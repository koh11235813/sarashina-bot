# Sarashina Discord Bot — 仕様書 v0.1（確定待ちなし・決定事項のみ）

## 概要

- Discord Bot 本体を Python 3.12 + discord.py で実装
- LLM は Sarashina API（`https://api.sarashina.ai/v1`）
- デプロイ先: AlmaLinux 上の k3s
- パッケージ管理: uv

## 決定事項

| 項目 | 決定 |
| --- | --- |
| LLM モデル | sarashina3-mini のみ（guard は不使用） |
| 会話履歴 | メンション時、直近 N 件のチャンネル履歴をコンテキストに含める |
| 応答方式 | 単発送信（生成完了後に 1 メッセージ送信。ストリーミング・編集なし） |
| LLM クライアント | 自前の薄いラッパー（httpx。OpenAI SDK は使わない） |
| 永続化 | 会話履歴は保存しない（Discord 側の履歴を都度取得） |

## Sarashina API 対応仕様（独自実装部分）

Chat Completions は OpenAI 互換に見えるが、**ドキュメント未記載のパラメータを
1 つでも送ると HTTP 400** で落ちる。クライアントは以下を守る。

- リクエスト JSON はホワイトリスト方式で構築する（必要なキーだけ明示的に積む）
- 出力上限は `max_completion_tokens`（既定・上限とも 8,000）。`max_tokens` は 400
- `logprobs` / `stream_options` 等は送らない
- `GET /v1/models` は存在しない（404）→ モデル ID は設定値（ConfigMap）
- `tool_choice` を `tools` と併用する場合、自律判断させたいなら `"auto"` を明示指定
  （省略時は `required` 扱い）。本仕様ではツール呼び出しは使用しない
- ストリーミング不使用のため `stream` は常に送らない
- エラー本文は `{"message": "..."}` のみ。400 の原因は body から分からないので、
  クライアントは送出リクエスト JSON 全体を debug ログに出す

### レート制限対応

- 無償ベータは 6 RPM。429 には `Retry-After` ヘッダーも残数ヘッダーも無い
- クライアント側に固定レートリミッタ（例: 1 分あたり 5 リクエストで止める）＋
  429 時の指数バックオフリトライ（上限 1 回）を実装
- リミッタはプロセス内のみでよい（単一レプリカ構成のため）

## Bot 動作仕様

- トリガー: メンション（`on_message` で `bot.user.mentioned_in()`）
- コンテキスト構築:
  1. メンションされたメッセージのチャンネルから直近 N 件（既定 10 件）を取得
     （`channel.history(limit=N)`、bot 自身の応答も含める）
  2. 各メッセージを `author 名: 内容` 形式で user ロールに整形
  3. メンション対象メッセージを最後の user 発言とする
  4. システムプロンプトを先頭に付与（内容は ConfigMap で差し替え可能）
- 応答: 生成完了後に単発で返信。長文は Discord 上限 2,000 文字で分割送信
- エラー時: API 失敗・レート制限超過は「返信に失敗した」旨の短文を返す
  （スタックトレース等を Discord に出さない）

## 設定・シークレット

| 項目 | 提供方法 |
| --- | --- |
| `DISCORD_TOKEN` | k3s Secret |
| `SARASHINA_API_KEY` | k3s Secret |
| `SARASHINA_MODEL`（既定 sarashina3-mini） | ConfigMap |
| `CONTEXT_MESSAGE_COUNT`（既定 10） | ConfigMap |
| `SYSTEM_PROMPT` | ConfigMap |
| `RATE_LIMIT_RPM`（既定 5） | ConfigMap |

## k3s デプロイ仕様

- kind: Deployment、`replicas: 1`、`strategy: Recreate`
  （discord.py は二重起動すると token 競合するため RollingUpdate 不可）
- リソース要求: ごく軽量（CPU 100m / Memory 128Mi 程度で足りる見込み）
- コンテナイメージ: uv でビルドしたマルチステージ Docker イメージ
  （python:3.12-slim ベース。実行ユーザーは non-root）
- ヘルスチェック: プロセス生存のみでよい（外部依存の liveness probe は付けない。
  Discord Gateway の再接続は discord.py が内包するため）

## 非目標（スコープ外）

- スラッシュコマンド、ボタン/選択 UI
- Embeddings / Rerank / Translate エンドポイントの利用
- 会話履歴のデータベース保存
- モデレーション（sarashina3-guard）

## 今後の検証事項（実装前に確認）

- [ ] 無償ベータの提供終了時期（終了時に有償プランへ移行するかの判定）
- [ ] レート制限の有償時の値（設計のレートリミッタは設定値化済みなので影響小）
