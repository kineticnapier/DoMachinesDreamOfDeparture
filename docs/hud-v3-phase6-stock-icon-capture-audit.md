# DMDOD HUD v3 Phase 6 — Stock SetSpeed icon capture audit

## 現在の判定

`UNVERIFIED`

撮影用8x/1x譜面と監査ツールは完成したが、paired video、crop、時刻同期、capture latencyがまだ入力されていない。したがって`OBSERVABLE_BEFORE_DEADLINE`を主張しない。

固定deadlineは **episode 3.9761634902576737s**。Timing、Fingering、Motor、Scorer、checkpointは変更していない。

## 撮影用譜面

生成済みローカルpackage:

`artifacts/setspeed_icon_capture_phase6/prepared-9cc299c-r2/`

| world | chart SHA256 | 音声 |
| --- | --- | --- |
| original-8x | `04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1` | 未同梱 |
| counterfactual-1x | `0864c36782c352bc739e8ae1577041536f7ab61b029922e5370579c4b55bd9c5` | 未同梱 |

元Train80のchartは変更していない。1xファイルは元bytes内の、唯一のactive Floor 1 `SetSpeed.bpmMultiplier` token `8`を`1`へ置換した。生成器はparsed treeを比較し、それ以外のsemantic differenceがあれば失敗する。

Train80には`bool.ogg`がないため、実際のlevel packageが参照する同一の`bool.ogg`をユーザー側で指定して新しいpackageを生成する。

~~~powershell
.\.venv\Scripts\python.exe scripts/prepare_stock_setspeed_icon_capture_v01.py `
  --audio-source "X:\path\to\original\bool.ogg" `
  --output-dir artifacts/setspeed_icon_capture_phase6/capture-ready-r2
~~~

出力先が存在する場合は上書きせず失敗する。両worldへ同じaudioをコピーし、SHA256を`package.json`へ記録する。

## 固定撮影手順

1. ADOFAI、解像度、UI scale、fullscreen/window mode、camera、VFX設定を固定する。
2. JRP/ShowBPMなどのModを無効にする。使用Modが残る場合は全名称とversionをmanifestへ記録する。
3. 8xと1xを別folderから開く。譜面設定をEditorで保存し直さない。
4. 可能ならCFR 120fps以上、同一codec・解像度・cropなしで、Countdown開始前から第2event後まで録画する。
5. Floor 1 tileとspeed iconの予想位置が画面内に入ることを確認する。ただし撮影中にcameraを動かさない。
6. 同一のfirst countdown tickを両動画のsync anchorとして、動画上の時刻を記録する。Phase 4 evaluatorではこのtickをepisode 0sとして扱う。
7. capture pipelineのdisplay-to-frame availability latencyを測り、単一点ではなく`min_s`/`max_s`を記録する。測定不能なら空欄のままにし、判定を`UNVERIFIED`に保つ。

60fpsでも監査は拒否しないが、frame half-widthは約8.33msになる。120fpsなら約4.17ms。いずれもsync uncertaintyとlatencyを加算した上限がdeadline以前である必要がある。

## Manifest

生成された`capture-manifest.template.json`を別名へコピーし、次を記入する。

- `video_path`: manifestからの相対pathまたは絶対path
- `recording`: game build、録画software、mode、reported FPS、resolution、Mod、camera/display条件
- `crop_xywh`: 両動画に共通する`[x, y, width, height]`
- `sync.anchor_video_s`: first countdown tickのvideo timestamp
- `sync.uncertainty_s`: anchor検出とA/V同期を含む不確実性
- `capture_latency.min_s/max_s/basis`
- `annotations`: frameを目視確認した時刻、状態、証拠画像path、必要なら`first_occurrence_confirmed`

使用可能な状態は次の3つだけ。

- `DOUBLE_RABBIT_VISIBLE`
- `TILE_VISIBLE_NO_SPEED_ICON`
- `UNKNOWN_OR_OCCLUDED`

1xの`TILE_VISIBLE_NO_SPEED_ICON`は、Floor 1 tileとspeed-icon描画位置の両方が判読できる場合だけ使用する。tileが画面外、暗転、filter、overlay、圧縮で不鮮明なら`UNKNOWN_OR_OCCLUDED`とする。単なる黒pixelや「見つからなかった」は1xの証拠にならない。

annotation例:

~~~json
{
  "video_time_s": 5.125,
  "state": "DOUBLE_RABBIT_VISIBLE",
  "evidence": "frames/original-8x-0615.png",
  "first_occurrence_confirmed": true
}
~~~

## 時刻変換と判定

各annotationについて監査器は次を計算する。

~~~text
episode_center = video_time - anchor_video + anchor_episode
frame_half_width = 0.5 / probed_fps
arrival_lower = episode_center - sync_uncertainty - frame_half_width + latency_min
arrival_upper = episode_center + sync_uncertainty + frame_half_width + latency_max
~~~

両worldを必ず識別できる時刻は、それぞれの最初の正しい状態の`arrival_upper`の大きい方である。

- 両方のupperがdeadline以前: `OBSERVABLE_BEFORE_DEADLINE`
- 両方が揃い、区別時刻のlowerさえdeadline後で、かつそれ以前の全frameを確認して両annotationが最初の出現だと記録済み: `TOO_LATE`
- 不確実性区間がdeadlineを跨ぐ、metadata不足、UNKNOWN、1x tile不可視: `UNVERIFIED`

`first_occurrence_confirmed`は、anchorから該当frameまでを欠落なくレビューした場合だけtrueにする。遅い1frameを任意に選んだだけでは`TOO_LATE`にならず、`UNVERIFIED`を維持する。

Target truthはdeadlineとの事後照合だけに使用する。Studentへ渡せる値は、実pixelから得た3状態、validity、observed timestampだけである。DLL icon enum、Floor ID、倍率、compiled speedを渡してはならない。

## 撮影前の入力検証強化

録画監査器のレビューで、manifest上のラベルだけで過剰な成功判定をしないよう
以下を追加した。

- 8x/1x各worldはちょうど1件（重複禁止）
- manifestに記載した解像度とffprobeの実解像度が一致
- 両動画の実解像度が一致
- 同期アンカーはfirst countdown tick / episode 0に固定
- cropは整数4値、画面内、最大1024px四方に限定
- 比較フレームが0件、または`--no-pixel-analysis`で比較省略なら
  `OBSERVABLE_BEFORE_DEADLINE`を出さず`UNVERIFIED`へ棄権

これらは監査器の**入力信頼性**を強化するもので、実映像の観測データを
代替するものではない。

## Frame差解析

manifest完成後に実行する。

~~~powershell
.\.venv\Scripts\python.exe scripts/audit_stock_setspeed_icon_capture_v01.py `
  --manifest artifacts/setspeed_icon_capture_phase6/capture-ready-r2/capture-manifest.json `
  --output reports/hud-v3-phase6-stock-icon-capture-YYYYMMDD-rN.json
~~~

監査器は`ffprobe`で実FPS・解像度・durationを取得し、`ffmpeg`で同一cropをraw RGBへdecodeする。sync anchorでepisode時刻を揃え、deadline以前の対応frameについて平均絶対RGB差、最大差、16超のchannel比率を記録する。差分値は候補frame探索専用であり、閾値によって`DOUBLE_RABBIT_VISIBLE`を自動的に捏造しない。最終判定は明示状態と保守的arrival intervalで行う。

動画がない現在の再現コマンド:

~~~powershell
.\.venv\Scripts\python.exe scripts/audit_stock_setspeed_icon_capture_v01.py `
  --manifest artifacts/setspeed_icon_capture_phase6/prepared-9cc299c-r2/capture-manifest.template.json `
  --output reports/hud-v3-phase6-stock-icon-capture-YYYYMMDD-rN.json `
  --no-pixel-analysis
~~~

現在のr2結果はpaired video不足を列挙して`UNVERIFIED`で終了する。監査器は両動画のgame build、capture software/mode、reported FPS、resolution、Mod一覧、camera/display条件の一致も要求する。新しい実captureが得られるまでTiming-only sensor実装へ進まない。
