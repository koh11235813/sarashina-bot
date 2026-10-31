# Sarashina Discord Bot — 仕様書 v0.1（確定待ちなし・決定事項のみ）

## 概要

- Discord Bot 本体を Python 3.12 + discord.py で実装
- LLM は Sarashina API（`https://api.sarashina.ai/v1`）
- デプロイ先: AlmaLinux 上の k3s
- パッケージ管理: uv

## 決定事項

| 項目 | 決定 |
| --- | --- |
| LLM モデル | sarashina3-mini（既定）・sarashina3-nano・sarashina3.2-mini-thinking を `/model` で切り替え（guard は不使用） |
| 会話履歴 | メンション時、直近 N 件のチャンネル履歴のうち、メンションした人の発言と bot のその人への返事だけをコンテキストに含める |
| 応答方式 | 単発送信（生成完了後に 1 メッセージ送信。ストリーミング・編集なし） |
| LLM クライアント | 自前の薄いラッパー（httpx。OpenAI SDK は使わない） |
| 永続化 | 会話履歴は保存しない（メンション時は Discord 側の履歴を都度取得）。論文の索引はプロセス内のメモリだけに保持 |

## Sarashina API 対応仕様（独自実装部分）

Chat Completions は OpenAI 互換に見えるが、**ドキュメント未記載のパラメータを
1 つでも送ると HTTP 400** で落ちる。クライアントは以下を守る。

- リクエスト JSON はホワイトリスト方式で構築する（必要なキーだけ明示的に積む）
- 出力上限は `max_completion_tokens`（既定・上限とも 8,000）。`max_tokens` は 400
- `logprobs` / `stream_options` 等は送らない
- `GET /v1/models` は存在しない（404）→ モデル ID は設定値（ConfigMap）
- `tool_choice` を `tools` と併用する場合、自律判断させたいなら `"auto"` を明示指定
  （省略時は `required` 扱い）
- ストリーミング不使用のため `stream` は常に送らない
- `reasoning_effort`（`low` / `medium` / `high`）を受け付けるのは sarashina3.2-mini-thinking だけで、
  ほかのモデルに送ると 400（`reasoning_effort is not supported for model ...`）。
  そのため送るのはリクエスト時のモデル名がちょうど `sarashina3.2-mini-thinking` のときだけにする
  （`-thinking` で終わるかでは判定しない。対応が分からないモデルで 400 を招くため）
- エラー本文は `{"message": "..."}` のみ。400 の原因は body から分からないので、
  クライアントは送出リクエスト JSON 全体を debug ログに出す

### レート制限対応

- 上限は契約とモデルごとに決まる（1 分あたりのリクエスト数）

  | モデル | RPM |
  | --- | --- |
  | sarashina3-mini | 30 |
  | sarashina3.2-mini-thinking | 30 |
  | sarashina3-nano | 60 |
  | sarashina3-guard | 6 |
  | Embeddings / Rerank | 60 |
  | Translate | 30 |

- 1 分ごとにリセットされる固定ウィンドウで、バーストの持ち越しは無い
- 超えると 429 が返り、`Retry-After` ヘッダーに次のウィンドウまでの秒数（整数、切り上げ）が付く。
  残数などほかのレート制限ヘッダーは無い
- クライアント側にスライディングウィンドウ（直近 60 秒）のリミッタを置く（`RATE_LIMIT_RPM`、既定 30）。
  スライディングウィンドウは同じ上限の固定ウィンドウより厳しいので、リミッタを通る限りどの 1 分間でも上限を超えない
- リミッタはプロセス内のみでよい（単一レプリカ構成のため）
- 翻訳（sarashina3-translate）はチャットのモデルとは別枠の 30 RPM なので、クライアント側にも翻訳専用のリミッタを
  別に置く（上限は同じ `RATE_LIMIT_RPM`）。翻訳がチャットの枠を食わず、チャットも翻訳の枠を食わない
- Embeddings には待機型の専用リミッタを置く（既定 55 RPM。公式上限 60 RPM に余裕を持たせる）。
  索引構築と質問の埋め込みで共有し、チャットや翻訳の枠は使わない。
  Rerank にはクライアント側のリミッタを置かず、サーバーの 429 を通常どおり扱う
- API クライアントは 429 を再試行しない。固定ウィンドウなので、次のウィンドウまで待たない再試行は同じく弾かれる。
  例外はバックグラウンドの索引構築だけで、埋め込みが 429 になったら `Retry-After` の秒数を待ち、
  同じ段落を合計 3 回まで試す。3 回目も失敗したら呼び出し元へ伝える
- リミッタで止めたときは必要な空きができるまでの秒数、429 のときは `Retry-After` の秒数
  （ヘッダーが無い・読めないときは 60 秒）を使い、ユーザーに「{x}秒後にもう一度送信してください。」と返す。
  ツールを回している途中で当たったときも、それまでの結果は捨てて同じ返事をする

## Bot 動作仕様

- メンション判定より先に、bot が所有するスレッドの論文処理と、テキストチャンネルの arXiv URL を判定する。
  それ以外のトリガーはメンション（`on_message` で `bot.user.mentioned_in()`）
- コンテキスト構築:
  1. メンションされたメッセージのチャンネルから直近 N 件（既定 10 件）を取得
     （`channel.history(limit=N)`、bot 自身の応答も含める）
  2. 取得した N 件を、メンションした人との会話だけに絞る
     - メンションした人の発言は残し、ほかの人（ほかの bot も含む）の発言は捨てる
     - bot 自身の発言は、メンションした人への返信（reply 先の author がその人）だけ残す
     - 長文を分割した 2 通目以降は reply にならないので、残した返信の直後に続く
       reply なしの bot 発言だけを続きとして残す。間にほかの発言が挟まったら続きとはみなさない
  3. メンションした人の発言は user ロール、bot の発言は assistant ロールにし、内容だけを入れる（名前は付けない）
  4. 先頭が assistant になる場合（問いかけが N 件の外に出た返信）は、user が出てくるまで捨てる。
     Sarashina のドキュメントはロールの並び順の制約を書いていないので、先頭を user にそろえておく
  5. メンション対象メッセージを最後の user 発言とする
  6. システムプロンプトを先頭に付与し、末尾に「話しかけているのは〇〇さんです。」を足す
     （システムプロンプトの内容は ConfigMap で差し替え可能）
- 応答: 生成完了後に単発で返信。長文は Discord 上限 2,000 文字で分割送信
- 最終回答の本文が空で `finish_reason` が `length` のとき（推論の途中で `max_completion_tokens` に達した）は
  「回答が出せませんでした。effort を下げてください。」と返す。ほかの理由で空のときは何も返さない
- エラー時: レート制限超過は「{x}秒後にもう一度送信してください。」、
  それ以外の API 失敗などは「返信に失敗した」旨の短文を返す
  （スタックトレース等を Discord に出さない）

## arXiv 論文のスレッドと RAG

- 誰でもテキストチャンネルに arXiv URL を貼ると、メンションなしで `arXiv:{id}` スレッドを作る。
  `arxiv.org`、`www.arxiv.org`、`export.arxiv.org` の `/abs/` と `/pdf/` を受け付け、
  バージョン付き ID と旧形式の ID にも対応する。他人のスレッドでは URL だけでは起動しない
- PDF を取得し（上限 50 MiB）、PyMuPDF でテキストを抽出する。
  抽出は `asyncio.to_thread` でイベントループの外に出す。
  ページ内のブロックをまとめ、長い段落は 1,800 文字以内に分割する。
  テキストのない PDF は短い失敗メッセージを返す
- 全段落の索引ができてから、各段落を `p.{ページ番号}` と本文からなる 1 メッセージとして順に投稿する。
  読み込みタスクは強参照で保持する。
  索引完成後は、段落を投稿している途中でも質問に答える
- bot が所有するスレッドでは、空でないメッセージを返信も含めてすべて論文への質問として扱う。
  メンションは不要で、ほかの bot の発言は無視する。`channel.history` は読まず、質問ごとに独立して回答する。
  索引構築中は待機を案内し、再起動などで索引を失ったら「もう一度 URL を貼ってください。」と案内する
- 質問を ja→en 翻訳し、埋め込みのコサイン類似度で上位 20 段落を選び、Rerank の上位 5 段落を使う。
  元の質問と抜粋だけを現在の `/model` のチャットモデルへ渡し、日本語で回答する。
  抜粋は外部データとして区切り、指示として扱わないよう指定する。
  回答にはページ引用と参考ページを付ける。ツールは渡さないため、ターミナルも呼べない。
  空回答はメンションと同じ扱いで、`finish_reason=length` なら effort を下げる案内を返す
- bot が投稿した段落を右クリック（モバイルでは長押し）→ アプリ → 日本語に翻訳で選ぶと、
  メッセージのコンテキストメニューコマンドで en→ja の訳文を返す。
  スラッシュコマンドは返信先のメッセージを受け取れず、返信中に `/translate` を入力しても
  必須引数付きの登録済みコマンドが開くため、段落翻訳には専用のコンテキストメニューを使う。
  段落以外やこの bot 以外のメッセージに使ったときは、本人にだけ段落を選ぶよう案内する
- 段落翻訳の権限は `/translate` と同じ `DISCORD_ALLOWED_USERS` に限定する。
  権限がない人には defer 前に本人にだけ「翻訳する権限がありません。」と返す。
  有効な段落は defer 後に翻訳し、訳文は全員に見える形で 2,000 文字ごとに分割して送る。
  レート制限、失敗、空の翻訳結果は `/translate` と同じ処理を使う。
  段落は選択したメッセージの著者がこの bot であることと形式から判定し、本文を直接取り出すため再起動後も翻訳できる。
  スレッド作成と RAG 質問は全員に開放する
- `POST /v1/embeddings` の本文は `{input, task_type, model: "sarashina3-embedding"}` のみ。
  `task_type` は索引に `RETRIEVAL_DOCUMENT`、質問に `RETRIEVAL_QUERY` を使う。
  `POST /v1/rerank` の本文は `{query, documents, task_type: "GENERAL_QA", model: "sarashina3-rerank"}` のみで、
  返された `document_id` の順序をそのまま使う
- 投稿と返信はメンションを無効にし、埋め込みを抑制する。
  予期しない失敗は固定のメッセージとスタックトレースをログに残す。
  レート制限では通常の再送待機メッセージ、その他の失敗では短い謝罪を返す

## ツール

- `web_search`（Exa）: 全員に見せる
- `run_command`（サンドボックスで bash を実行）: `DISCORD_ALLOWED_USERS` の人がメンションしたときだけ見せる。
  見せていないのにモデルが呼んだら実行せずにエラー文を返す
- 1 メンションで最大 `MAX_TOOL_ROUNDS` 回ツールを回し、最後は `tools` を外してテキスト回答を強制する。
  LLM 呼び出しは最大 `MAX_TOOL_ROUNDS + 1` 回で、最初の呼び出しでその分の空きをレートリミッタに確認する

## サンドボックス

- bot と同じイメージを `python -m sarashina_bot.sandbox` で別 Pod として動かす。Secret は渡さない
- `POST /run` に `{"command": str}` を受け、`bash -c` で実行して `{"exit_code", "timed_out", "output"}` を返す
- 実行ユーザーは non-root（UID 10001）、作業ディレクトリは `/work`（emptyDir。Pod が作り直されると消える）
- 30 秒で打ち切り、プロセスグループごと SIGKILL する（バックグラウンドのプロセスも残らない）
- 出力は stdout と stderr をまとめ、先頭 4,000 バイトまで返す
- 環境変数は PATH・HOME・LANG・TMPDIR だけ。ServiceAccount トークンはマウントしない
- NetworkPolicy: ingress は bot の Pod からだけ。egress は DNS とインターネットだけ（LAN・ホスト・cluster 内へは出さない）

## アプリコマンド

- `/model [name]`: `SARASHINA_MODELS` から選んで切り替える。省略すると今のモデルを表示する。
  `DISCORD_ALLOWED_USERS` の人だけ使え、応答は本人にだけ見える（ephemeral）
- `/effort level`: `low` / `medium` / `high` から選んで `reasoning_effort` を指定する。
  今のモデルが sarashina3.2-mini-thinking でなければ「{model} では effort を指定できません。」と返して何も変えない。
  使える人と ephemeral は `/model` と同じ。起動直後は未指定で、そのあいだは `reasoning_effort` を送らない
- 指定した effort は `/model` で切り替えても残り、sarashina3.2-mini-thinking に戻すとまた効く
- 切り替えはプロセス内だけで、Pod が作り直されると `SARASHINA_MODEL` と effort 未指定に戻る
- `/translate source text`: `source`（`ja` / `en`）の文章 `text` をもう一方の言語に翻訳する（`POST /v1/translate`）。
  `text` は 1〜5,000 文字で、超える入力は Discord 側で弾く。使える人は `/model` と同じで、
  ほかの人には「翻訳する権限がありません。」と本人にだけ見える返事をする。
  翻訳は 3 秒を超えうるので defer してから返し、返事は全員に見える形で翻訳文だけを送る（原文は付けない）。
  2,000 文字を超えたら分割して送る。レート制限は「{x}秒後にもう一度送信してください。」、
  それ以外の失敗は「翻訳に失敗しました。」と返す（API のエラー内容は Discord に出さずログに残す）。
  ログに残すのは元の言語と文字数だけで、文章そのものは出さない。翻訳は会話履歴にもモデルのツールにも入れない
- メッセージのコンテキストメニュー「日本語に翻訳」: bot が投稿した論文の段落を翻訳する。
  操作と権限は「arXiv 論文のスレッドと RAG」に記載
- コマンドは起動のたびに `tree.sync()` でグローバルに登録する

## 設定・シークレット

| 項目 | 提供方法 |
| --- | --- |
| `DISCORD_TOKEN` | k3s Secret |
| `SARASHINA_API_KEY` | k3s Secret |
| `SARASHINA_MODEL`（既定 sarashina3-mini） | ConfigMap |
| `SARASHINA_MODELS`（`/model` の選択肢、カンマ区切り。`SARASHINA_MODEL` を含める） | ConfigMap |
| `DISCORD_ALLOWED_USERS`（ターミナルと `/model`・`/effort`・`/translate`・「日本語に翻訳」を使える Discord ユーザー ID、カンマ区切り） | k3s Secret |
| `SANDBOX_URL`（既定 `http://sandbox:8080`） | ConfigMap |
| `MAX_TOOL_ROUNDS`（既定 2。`+ 1` が `RATE_LIMIT_RPM` 以下） | ConfigMap |
| `CONTEXT_MESSAGE_COUNT`（既定 10） | ConfigMap |
| `SYSTEM_PROMPT` | ConfigMap |
| `RATE_LIMIT_RPM`（既定 30。sarashina3-mini・sarashina3.2-mini-thinking の上限。翻訳のリミッタにも別枠で同じ値を使う） | ConfigMap |

## k3s デプロイ仕様

- kind: Deployment、`replicas: 1`、`strategy: Recreate`
  （discord.py は二重起動すると token 競合するため RollingUpdate 不可）
- リソース要求: ごく軽量（CPU 100m / Memory 128Mi 程度で足りる見込み）
- コンテナイメージ: uv でビルドしたマルチステージ Docker イメージ
  （python:3.12-slim ベース。実行ユーザーは non-root）
- ヘルスチェック: プロセス生存のみでよい（外部依存の liveness probe は付けない。
  Discord Gateway の再接続は discord.py が内包するため）

## 非目標（スコープ外）

- `/model`・`/effort`・`/translate` 以外のスラッシュコマンド、「日本語に翻訳」以外のコンテキストメニュー、ボタン/選択 UI
- arXiv 以外の PDF サイトへの対応と OCR
- 論文索引の永続化と再起動後の自動復元
- 会話履歴のデータベース保存
- モデレーション（sarashina3-guard）

## 今後の検証事項（実装前に確認）

- [ ] 無償ベータの提供終了時期（終了時に有償プランへ移行するかの判定）
- [x] レート制限の値（契約とモデルごと。「レート制限対応」に記載）
