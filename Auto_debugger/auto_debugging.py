#!/usr/bin/env python3
"""
汎用的な Staff-Customer 会話練習システム（AutoGen + Anthropic/Gemini）- シンプルログ版

必要なファイル:
- customer_persona.md: 顧客ペルソナの定義（必須）- テスト対象のLLM設定
- staff_persona.md: 店員/スタッフペルソナの定義（必須）
- evaluator_prompt.md: 評価エージェント用プロンプト（必須）
- .env: ANTHROPIC_API_KEY=your_api_key_here または GOOGLE_API_KEY=your_api_key_here
"""

import os
import json
from pathlib import Path
from datetime import datetime
import logging
from dotenv import load_dotenv
import asyncio
import traceback

# AutoGen関連のインポート
from autogen_core.models import UserMessage, ModelInfo # ModelInfo をインポート
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

# 環境変数の読み込み
load_dotenv()

class ConversationTestingSystem:
    def __init__(self):
        self.setup_logging()
        self.setup_config() # APIキーのチェックをここで行う
        self.load_all_prompts()
        self.loop = asyncio.get_event_loop() # 既に存在する場合はそれを取得
        if self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def setup_logging(self):
        """ログ設定の初期化"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = log_dir / f"conversation_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # レベル名追加
            handlers=[
                logging.FileHandler(self.log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ConversationTest")

        self.conversation_log_file = log_dir / f"chat_{timestamp}.json"
        self.conversation_log = []

    def setup_config(self):
        """API設定の初期化（Anthropic/Gemini対応）"""
        self.api_provider = os.getenv('API_PROVIDER', 'anthropic').lower()
        self.customer_model_client = None
        self.staff_model_client = None

        if self.api_provider == 'gemini':
            self._setup_gemini()
        elif self.api_provider == 'anthropic':
            self._setup_anthropic()
        else:
            raise ValueError(f"Unsupported API_PROVIDER: {self.api_provider}. Choose 'anthropic' or 'gemini'.")

        self.logger.info(f"API Provider: {self.api_provider}")

    def _setup_anthropic(self):
        """Anthropic API設定"""
        try:
            from autogen_ext.models.anthropic import AnthropicChatCompletionClient
        except ImportError:
            self.logger.error("AnthropicChatCompletionClient のインポートに失敗しました。'autogen-ext[anthropic]' がインストールされているか確認してください。")
            raise ImportError("必要なパッケージがインストールされていません。pip install 'autogen-ext[anthropic]' を実行してください")

        claude_key = os.getenv('ANTHROPIC_API_KEY')
        if not claude_key:
            raise ValueError("ANTHROPIC_API_KEY環境変数が設定されていません。")

        self.customer_model_client = AnthropicChatCompletionClient(
            model=os.getenv('ANTHROPIC_CUSTOMER_MODEL', "claude-3-haiku-20240307"),
            api_key=claude_key,
        )
        self.logger.info(f"Customer (Anthropic) model: {self.customer_model_client.model}")

        self.staff_model_client = AnthropicChatCompletionClient(
            model=os.getenv('ANTHROPIC_STAFF_MODEL', "claude-3-5-sonnet-20240620"),
            api_key=claude_key,
        )
        self.logger.info(f"Staff/Evaluator (Anthropic) model: {self.staff_model_client.model}")

    def _setup_gemini(self):
        """Google Gemini API設定（OpenAI互換API使用）"""
        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            # ModelInfo は autogen_core.models から既にインポート済み
        except ImportError:
            self.logger.error("OpenAIChatCompletionClient のインポートに失敗しました。'autogen-ext[openai]' がインストールされているか確認してください。")
            raise ImportError("必要なパッケージがインストールされていません。pip install 'autogen-ext[openai]' を実行してください")

        google_key = os.getenv('GOOGLE_API_KEY')
        if not google_key:
            raise ValueError("GOOGLE_API_KEY環境変数が設定されていません。")

        customer_model_name = os.getenv('GEMINI_CUSTOMER_MODEL', "gemini-1.5-flash-latest")
        staff_model_name = os.getenv('GEMINI_STAFF_MODEL', "gemini-1.5-flash-latest")

        model_info_common = ModelInfo(
            family="gemini",
            vision=True,
            function_calling=True,
            json_output=True,
        )

        gemini_api_base_url = os.getenv('GEMINI_API_BASE_URL', "https://generativelanguage.googleapis.com/v1beta")

        self.customer_model_client = OpenAIChatCompletionClient(
            model=customer_model_name,
            api_key=google_key,
            base_url=gemini_api_base_url,
            model_info=model_info_common,
            api_type="google",
        )
        self.logger.info(f"Customer (Gemini) model: {customer_model_name}, Client Base URL: {gemini_api_base_url}")

        self.staff_model_client = OpenAIChatCompletionClient(
            model=staff_model_name,
            api_key=google_key,
            base_url=gemini_api_base_url,
            model_info=model_info_common,
            api_type="google",
        )
        self.logger.info(f"Staff/Evaluator (Gemini) model: {staff_model_name}, Client Base URL: {gemini_api_base_url}")

    def load_file_content(self, file_path: Path, description: str) -> str:
        """ファイル内容を読み込む汎用メソッド"""
        if not file_path.exists():
            self.logger.error(f"{description}ファイルが見つかりません: {file_path}")
            raise FileNotFoundError(f"{description}ファイルが見つかりません: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                self.logger.warning(f"{description}ファイルが空です: {file_path}")
            self.logger.info(f"{description}を読み込みました: {file_path}")
            return content
        except Exception as e:
            self.logger.error(f"{description}ファイルの読み込みエラー: {e}", exc_info=True)
            raise Exception(f"{description}ファイルの読み込みエラー: {e}")

    def load_all_prompts(self):
        """全てのプロンプトファイルを読み込み"""
        self.customer_persona = self.load_file_content(
            Path("customer_persona.md"), "顧客ペルソナ"
        )
        self.staff_persona = self.load_file_content(
            Path("staff_persona.md"), "スタッフペルソナ"
        )
        self.evaluator_prompt = self.load_file_content(
            Path("evaluator_prompt.md"), "評価プロンプト"
        )
        if not (self.customer_persona and self.staff_persona and self.evaluator_prompt):
            raise ValueError("必須プロンプトファイル (customer_persona.md, staff_persona.md, evaluator_prompt.md) のいずれかが空または読み込めませんでした。")

    def log_conversation(self, speaker: str, content: str):
        """会話内容をログに記録"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "speaker": speaker,
            "content": content
        }
        self.conversation_log.append(log_entry)

        try:
            with open(self.conversation_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"会話ログ保存エラー: {e}", exc_info=True)

        self.logger.info(f"Chat Logged - [{speaker}]: {content[:100]}{'...' if len(content) > 100 else ''}")

    async def create_agents(self, scenario_description: str = "", max_turns: int = 10):
        """エージェントの作成（非同期）"""

        conversation_guidelines = f"""
        ### 会話のガイドライン
        - 1回の発言は簡潔に（2-4文程度）
        - 自然な会話の流れを意識
        - 同じ内容を繰り返さない
        - これはロールプレイであり、研修や振り返りの話は直接行わないでください。
        - 最大{max_turns // 2}回程度のやり取りで会話の目的を達成することを目指してください。

        ### 会話終了条件
        - 会話の目的が達成され、これ以上話すことがないと判断した場合、あなたの発言の最後に「DONE」という単語を必ず含めてください。
        - 例: 「ありがとうございました。これで解決しました。DONE」
        - 「DONE」の後には、いかなる追加のテキストも出力しないでください。
        - 質問や応答が残っている場合、または相手の応答を待つべき場合は「DONE」を出力しないでください。

        ### 現在のシナリオ
        {scenario_description}

        ### 重要な注意事項
        - あなたは指定されたペルソナになりきってください。
        - 直前の相手の発言で会話が終了条件を満たしている場合でも、あなた自身の応答として「DONE」と出力してください（例：相手が「ありがとうございました。DONE」と言ったら、あなたも「こちらこそありがとうございました。DONE」のように）。ただし、不自然な場合は無理に「DONE」を繰り返す必要はありません。
        - 「お疲れ様でした」のようなメタ的な発言は避けてください。
        """

        customer_system_message = f"{self.customer_persona}\n{conversation_guidelines}"
        self.customer_agent = AssistantAgent(
            name="Customer",
            description="顧客/クライアント役。テスト対象のLLM。",
            system_message=customer_system_message,
            model_client=self.customer_model_client,
        )

        staff_system_message = f"{self.staff_persona}\n{conversation_guidelines}"
        self.staff_agent = AssistantAgent(
            name="Staff",
            description="スタッフ/サービス提供者役。",
            system_message=staff_system_message,
            model_client=self.staff_model_client,
        )

        self.evaluator_agent = AssistantAgent(
            name="Evaluator",
            description="会話品質とペルソナ一貫性を評価する専門家。",
            system_message=self.evaluator_prompt,
            model_client=self.staff_model_client,
        )
        self.logger.info("All agents created successfully.")

    async def run_conversation_test(self, scenario_description: str = "一般的な会話", initial_message_content: str = "こんにちは", max_turns: int = 10):
        """会話テストの実行（非同期）- シンプル版"""

        print("\n" + "=" * 80)
        print("🎭 会話テストシステム 開始")
        print("=" * 80)
        print(f"シナリオ: {scenario_description}")
        print(f"最大ターン数: {max_turns}")
        print(f"ログファイル (詳細): {self.log_filename}")
        print(f"会話ログ (JSON): {self.conversation_log_file}")
        print("-" * 80)

        self.logger.info(f"Test session started: Scenario - '{scenario_description}', Initial Message - '{initial_message_content}', Max Turns - {max_turns}")
        self.log_conversation("System", f"シナリオ: {scenario_description}")

        all_final_messages = []

        try:
            await self.create_agents(scenario_description, max_turns)

            print("\n🎬 会話開始...")
            print("=" * 40)

            agents_for_chat = [self.customer_agent, self.staff_agent]

            group_chat = RoundRobinGroupChat(
                participants=agents_for_chat,
                max_turns=max_turns,
                termination_condition=TextMentionTermination('DONE')
            )

            chat_result = await group_chat.run(
                task=scenario_description
            )

            print("\n📜 会話履歴:")
            print("=" * 40)

            if chat_result and hasattr(chat_result, 'messages') and chat_result.messages:
                conversation_history = chat_result.messages
            else:
                self.logger.warning("No conversation history found in chat_result or group_chat.messages.")

            for i, msg_obj in enumerate(conversation_history):
                speaker = "Unknown"
                content = ""

                if isinstance(msg_obj, dict) and "content" in msg_obj and isinstance(msg_obj["content"], str):
                    content = msg_obj["content"]
                    # 'name' または 'source' があればそれを speaker とする
                    if "name" in msg_obj and msg_obj["name"]:
                        speaker = msg_obj["name"]
                    elif "source" in msg_obj and msg_obj["source"]: # 追加
                        speaker = msg_obj["source"]
                    elif "role" in msg_obj and msg_obj["role"]:
                         speaker = msg_obj["role"].capitalize()
                elif hasattr(msg_obj, 'content') and isinstance(msg_obj.content, str):
                    content = msg_obj.content
                    if hasattr(msg_obj, 'name') and msg_obj.name:
                        speaker = msg_obj.name
                    elif hasattr(msg_obj, 'source') and msg_obj.source: # 追加
                        speaker = msg_obj.source
                    elif hasattr(msg_obj, 'role') and msg_obj.role:
                        speaker = msg_obj.role.capitalize()
                else:
                    content = str(msg_obj)
                    self.logger.warning(f"Message object at index {i} has no 'content' or is not a string: {msg_obj}")

                # sender 属性からのフォールバック (必要であれば)
                if speaker == "Unknown" and hasattr(msg_obj, 'sender') and msg_obj.sender and hasattr(msg_obj.sender, 'name'):
                     speaker = msg_obj.sender.name

                if speaker == "User_CLI_Input":
                    speaker = "Customer"

                all_final_messages.append({"source": speaker, "content": content})
                self.log_conversation(speaker, content)
                print(f"\n[{speaker}]: {content}")
                print("-" * 40)

            conversation_ended_naturally = False
            if all_final_messages and "DONE" in all_final_messages[-1]['content'].strip().upper():
                conversation_ended_naturally = True

            total_turns = len(all_final_messages)

            print("\n🎬 会話終了")
            if conversation_ended_naturally:
                self.logger.info("Conversation ended naturally (DONE detected).")
                print("（エージェントの指示により自然な終了を検知しました）")
            else:
                self.logger.info(f"Conversation ended due to max_turns ({max_turns}) or other reasons.")
                print(f"（最大ターン数 {max_turns} に到達したか、他の理由で終了しました）")
            print("=" * 40)

            print("\n📊 評価開始...")
            self.logger.info("Starting evaluation phase.")

            conversation_summary_for_eval = "\n".join([
                f"[{msg_item['source']}]: {msg_item['content']}"
                for msg_item in all_final_messages
            ])

            evaluation_input_text = f"""
            以下の会話ログ全体を評価してください。

            ### 評価対象の会話
            {conversation_summary_for_eval}
            """

            evaluation_response_message_obj = await self.evaluator_agent.run(
                task=evaluation_input_text
            )

            if evaluation_response_message_obj:
                if hasattr(evaluation_response_message_obj, 'messages'):
                    evaluation_content = evaluation_response_message_obj.messages[-1].content
                else:
                    evaluation_content = str(evaluation_response_message_obj)
                    self.logger.warning(f"Unexpected evaluation response format: {type(evaluation_response_message_obj)}")
            else:
                self.logger.error("Evaluation agent did not return a response.")
                evaluation_content = "評価エラー: 評価エージェントから応答がありませんでした。"

            self.log_conversation("Evaluator", evaluation_content)

            print("\n" + "=" * 80)
            print("📝 評価結果")
            print("=" * 80)
            print(evaluation_content)
            print("=" * 80)

            self.logger.info("テストセッション完了")
            
            # クライアントをクローズ
            await self.customer_model_client.close()
            await self.staff_model_client.close()

            return {
                "chat_messages": all_final_messages,
                "evaluation": evaluation_content,
                "log_file_json": str(self.conversation_log_file),
                "log_file_system": str(self.log_filename),
                "conversation_ended_naturally": conversation_ended_naturally,
                "total_turns": total_turns
            }

        except Exception as e:
            error_msg = f"テスト中にエラーが発生しました: {e}"
            self.logger.error(error_msg, exc_info=True)
            print(f"\n❌ {error_msg}")
            traceback.print_exc()
            return None
        finally:
            self.logger.info("Attempting to close LLM clients.")
            try:
                if self.customer_model_client and hasattr(self.customer_model_client, 'close'):
                    await self.customer_model_client.close()
                    self.logger.info("Customer model client closed.")
            except Exception as ce:
                self.logger.error(f"Error closing customer model client: {ce}", exc_info=True)
            try:
                if self.staff_model_client and hasattr(self.staff_model_client, 'close'):
                    await self.staff_model_client.close()
                    self.logger.info("Staff model client closed.")
            except Exception as se:
                self.logger.error(f"Error closing staff model client: {se}", exc_info=True)

    def get_conversation_stats(self) -> dict:
        """会話統計の取得"""
        if not self.conversation_log:
            return {}

        customer_msgs = len([msg for msg in self.conversation_log if msg["speaker"].lower() == "customer"])
        staff_msgs = len([msg for msg in self.conversation_log if msg["speaker"].lower() == "staff"])

        stats = {
            "total_logged_entries": len(self.conversation_log),
            "customer_messages": customer_msgs,
            "staff_messages": staff_msgs,
            "actual_conversation_turns": customer_msgs + staff_msgs,
            "duration_seconds": None,
            "duration_formatted": "N/A"
        }

        conversation_entries = [
            msg for msg in self.conversation_log
            if msg["speaker"].lower() in ["customer", "staff", "user_cli_input", "user", "system"] # "System"も開始点として考慮
        ]

        if len(conversation_entries) >= 1: # 最初のメッセージだけでも時間は計算できる（ほぼ0だが）
            try:
                # 最初の意味のあるログエントリ（Systemのシナリオ設定や最初のユーザー発言）
                first_entry_time_str = conversation_entries[0]["timestamp"]
                # 最後の意味のあるログエントリ (評価結果の一つ前、つまり会話の最後)
                # "Evaluator" を除く最後のメッセージ
                last_conv_entries = [m for m in conversation_entries if m["speaker"].lower() != "evaluator"]
                if not last_conv_entries: # 会話が全く無かった場合
                    last_entry_time_str = first_entry_time_str
                else:
                    last_entry_time_str = last_conv_entries[-1]["timestamp"]

                start_time = datetime.fromisoformat(first_entry_time_str)
                end_time = datetime.fromisoformat(last_entry_time_str)
                duration_delta = end_time - start_time
                stats["duration_seconds"] = duration_delta.total_seconds()
                stats["duration_formatted"] = str(duration_delta)
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Could not parse timestamps for duration calculation: {e}")

        return stats

async def main():
    """メイン実行関数（非同期）"""
    test_system = None
    try:
        print("🔧 システム初期化中...")
        test_system = ConversationTestingSystem()
        print("✅ 初期化完了 (ログ・API設定)")
        print("✅ 全プロンプトファイル読み込み完了")

        print("\n📋 会話テストの設定")
        print("-" * 40)

        scenario = input("シナリオを入力してください（空白でデフォルト「一般的なカスタマーサポート」）: ").strip()
        if not scenario:
            scenario = "一般的なカスタマーサポートの会話"

        initial_msg_content = input("最初のメッセージを入力してください（空白でデフォルト「こんにちは」）: ").strip()
        if not initial_msg_content:
            initial_msg_content = "こんにちは。"

        max_turns=100

        print(f"\n🎭 テスト設定:")
        print(f"  シナリオ: {scenario}")
        print(f"  初期メッセージ: {initial_msg_content}")
        print(f"  最大ターン数: {max_turns}")

        result = await test_system.run_conversation_test(
            scenario_description=scenario,
            initial_message_content=initial_msg_content,
            max_turns=max_turns
        )

        if result:
            stats = test_system.get_conversation_stats()
            if stats:
                print(f"\n📊 会話統計:")
                print(f"  記録された総ログエントリ数: {stats.get('total_logged_entries', 'N/A')}")
                print(f"  顧客発言数: {stats.get('customer_messages', 'N/A')}")
                print(f"  スタッフ発言数: {stats.get('staff_messages', 'N/A')}")
                print(f"  実際の会話ターン数 (顧客+スタッフ): {stats.get('actual_conversation_turns', 'N/A')}")
                if stats.get('duration_formatted') != "N/A":
                    print(f"  会話時間 (概算): {stats['duration_formatted']}")

            print(f"\n💾 システムログファイル: {result.get('log_file_system', 'N/A')}")
            print(f"💾 会話JSONログファイル: {result.get('log_file_json', 'N/A')}")
            print("\n🎉 テスト完了！")
        else:
            print("\n❌ テストセッションは結果を返さずに終了しました。詳細はログを確認してください。")

    except FileNotFoundError as e:
        print(f"\n❌ ファイルエラー: {e}")
        print("プログラムを続行できません。必要なファイルを確認してください。")
        print("\n📝 必要なファイル:")
        print("  - customer_persona.md")
        print("  - staff_persona.md")
        print("  - evaluator_prompt.md")
        print("  - .env (APIキー設定: ANTHROPIC_API_KEY または GOOGLE_API_KEY)")
    except ValueError as e:
        print(f"\n❌ 設定エラー: {e}")
        print("プログラムを続行できません。設定を確認してください。")
        print("\n🔧 必要な設定:")
        print("  1. .env ファイルに ANTHROPIC_API_KEY (または GOOGLE_API_KEY) を設定")
        print("  2. API_PROVIDER 環境変数が 'anthropic' または 'gemini' に設定されていることを確認")
        print("  3. 全ての必須プロンプトファイル (customer_persona.md, staff_persona.md, evaluator_prompt.md) に内容を記述")
    except ImportError as e:
        print(f"\n❌ ライブラリエラー: {e}")
        print("必要なPythonライブラリがインストールされていない可能性があります。")
        print("インストールコマンドの例:")
        print("  pip install python-dotenv autogen-core 'autogen-agentchat>=0.2.20' 'autogen-ext[anthropic]' 'autogen-ext[openai]'")
    except Exception as e:
        print(f"\n❌ 予期せぬシステムエラーが発生しました: {e}")
        print("詳細はログファイルを確認してください。")
        traceback.print_exc()
    finally:
        print("\nシステムを終了します。")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nユーザーにより処理が中断されました。")
    except Exception as e:
        print(f"メイン実行中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()