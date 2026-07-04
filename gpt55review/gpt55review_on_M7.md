結論から言うと、M7は止めてよいです。
「エポック1が2時間以上終わらない」は、単なる我慢不足ではなく、SPEC上のCPUプロトタイプ目標から外れています。SPECではCPUプロトタイプは「1ラン60分以内」が目安で、M7はその段階的拡張実験のはずです。 ￼

1. プロジェクト評価

このプロジェクトは、方向性としてはかなり良いです。特に良い点は3つあります。

第一に、最初からフルスケールを狙わず、設計層とプロトタイプ層を分けている点です。SPECでは最大5変数・標準演算子集合を設計層に置き、CPUでは1〜3変数の縮小構成でend-to-end検証する方針になっています。これは正しいです。 ￼

第二に、M4〜M6で「ちゃんと失敗を踏んでいる」のが良いです。無条件拡散は最初パース成功率0.37で不合格、その後8000式×20エポックで0.8ゲートを通過しています。M5ではtiny1varで7/10、M6ではbest-of-kを64にすると9/10まで改善しています。これは、モデル単体では弱いが、候補生成＋BFGS＋選択でSRパイプラインとして成立することを示しています。 ￼

第三に、失敗要因の記録が良いです。PySR導入失敗、SRSD-Feynman取得失敗、arXiv遮断、損失重みの簡略化、式サンプリングの簡略化などが明記されています。これは研究プロトタイプとしてかなり重要です。 ￼

ただし、現状の評価はこうです。

M6までは「概念実証として有望」。M7は「一気にスケールしすぎて、CPU実験として破綻しかけている」。

2. M7で何が起きているか

M7の設定は、M5から見てかなり大きく跳ねています。

項目	M5	M7
変数数	1	3
演算子	add/sub/mul/sin	10種
系列長	L=16	L=32
d_model	64	128
データ数	20,000	80,000
エポック	8	6
エンコーダ	軽量構成と推定	PMAエンコーダ
学習時間	約3分	epoch 1が2時間超

これは「4倍のデータ」だけではありません。
語彙・式空間・系列長・モデル幅・条件付けエンコーダの計算量が同時に増えています。Transformer系では、ざっくり言うと d_model を64→128にすると、層内の行列演算は単純に2倍ではなく、複数箇所で効いてきます。さらに条件点数が100〜200点ある場合、PMA/Set Transformer風エンコーダがCPUで重くなります。

したがって、M7は「M6の自然な次」ではなく、複数の難化要因を同時投入したジャンプ実験になっています。

3. まずやるべき判断

M7の現在の実行は止める。

理由は明確です。

M7はまだ「proto3varの成立確認」であり、長時間回す段階ではありません。M5で約3分だったものが、M7で1エポック2時間超になっているなら、6エポック完走には半日〜1日級になります。しかも完走しても、設計上のどの要因が効いたのか分かりません。

ここで必要なのは完走ではなく、ボトルネック同定と段階的スケールアップです。

4. 直近の対処案

A. M7を分解する

M7をいきなり proto3var 本番で回すのではなく、以下の階段に分けるべきです。

実験	目的	推奨設定
M7a	3変数化だけを見る	3変数、演算子4種、L=16 or 24、d_model=64、20k式
M7b	系列長だけ伸ばす	3変数、演算子4種、L=32、d_model=64、20k式
M7c	演算子を増やす	3変数、演算子6〜7種、L=32、d_model=64、20k式
M7d	d_modelを増やす	d_model=128、ただしデータ20k
M7e	データを増やす	80k式、ただし他を固定

いまのM7は、この全部を一度にやっています。これは評価不能です。

B. まず「M7-lite」を作る

おすすめはこれです。

proto3var_lite:
  variables: 3
  operators: add, sub, mul, sin, cos
  L: 24
  d_model: 64
  encoder: mean_pool or simple_transformer
  n_points: 64
  dataset_size: 20,000
  epochs: 2
  batch_size: 64 or 128
  k_eval: 16

これで1エポックが10〜20分を超えるなら、実装にボトルネックがあります。
逆にこれが回るなら、M7本体が重すぎただけです。

C. PMAエンコーダを一度外す

M7ではPMAエンコーダが入っています。SPECではSet Transformer風の順序不変エンコーダが設計されていますが、同時に「プロトタイプでは通常のTransformerエンコーダ＋平均プーリングの簡易版から始めてもよい」とされています。 ￼

なので、M7ではまずPMAを外してよいです。

優先順位はこうです。

1. mean_pool_encoder
2. simple_transformer_encoder
3. PMA_encoder

PMAは性能改善用であって、最初の3変数PoCの必須部品ではありません。

5. 追加で必要なファイル

現時点で添付されているのはSPECと実験ログなので、設計評価はできますが、M7がなぜ遅いかの実装診断はまだできません。追加で必要なのは以下です。

必須

* diffsr/model/diffusion.py
* diffsr/model/encoder.py
* diffsr/model/decoder.py または Transformer本体
* diffsr/train.py または学習ループ
* diffsr/data/dataset.py、tree.py、式生成・評価点生成まわり
* M7の設定ファイル、例：configs/proto3var.yaml
* M7を実行したコマンド
* M7のログ、少なくとも以下：
    * batchごとのloss
    * epoch開始時刻
    * 何ステップ目まで進んだか
    * batch size
    * n_points
    * DataLoader num_workers
    * CPU使用率

あると非常に良い

* pip list
* pytest結果
* README.md
* pyproject.toml or requirements.txt
* runs/配下のログ
* time per batch が出ているログ
* 学習中に保存されたチェックポイント

6. M7が遅い原因候補

実装を見ないと断定はできませんが、可能性が高い順に並べるとこうです。

原因候補	典型症状	対処
DataLoader内でSymPy評価・式生成を毎回している	CPUが詰まり、GPUなし環境で極端に遅い	事前生成キャッシュ化
PMA/Set Transformerが重い	forwardだけで遅い	mean poolingに戻す
n_pointsが100〜200で大きい	encoder計算が支配的	32〜64点に減らす
batch sizeが小さすぎる	Python overheadが支配的	64/128に調整
PyTorch CPUスレッド設定が悪い	CPU使用率が低い	torch.set_num_threads(4)
評価処理がepoch中に走っている	epoch終盤で固まる	evaluationを分離
tqdm/logが更新されないだけ	実は進んでいるが見えない	batch単位ログを出す
サンプル生成・BFGSが学習中に混入	1 epochが異常に長い	学習と評価を完全分離

特に疑うべきは、学習ループの中に、式生成・SymPy・BFGS・サンプリング評価のどれかが混ざっていないかです。
M7は「学習」だけなら、重いとはいえ1エポック2時間超はやや不自然です。

7. すぐ入れるべき診断コード

M7を再実行する前に、学習ループにこれを入れるべきです。

import time
start_epoch = time.time()
for step, batch in enumerate(loader):
    t0 = time.time()
    # batch transfer / preprocessing
    t_data_ready = time.time()
    loss = train_step(batch)
    t_train_done = time.time()
    if step % 10 == 0:
        print(
            f"step={step} "
            f"data={t_data_ready - t0:.3f}s "
            f"train={t_train_done - t_data_ready:.3f}s "
            f"total={t_train_done - t0:.3f}s "
            f"loss={float(loss):.4f}",
            flush=True,
        )
print("epoch_time", time.time() - start_epoch)

これで、遅いのが

* データ読み込みか
* forward/backwardか
* 評価処理か
* ログが出ていないだけか

が分かります。

8. プロジェクト上の重要な論点

M6までの結果から、すでに大事なことが見えています。

この手法は、1ショットで正解式を出すモデルではない。
むしろ、拡散モデルは「候補スケルトンの多様な発生器」であり、その後段にBFGSと候補選択を置くことで機能しています。M6ではk=1だと1/10、k=64で9/10に伸びています。これはかなり本質的です。 ￼

したがって、今後の評価では「モデル精度」だけでなく、

候補生成器としての質
× 有効構文率
× best-of-kでの到達率
× BFGS後の選択能力

として見るべきです。

逆に言うと、M7で語彙を増やし、式空間を広げると、k=16やk=32では足りなくなる可能性が高いです。
つまりM7は、学習が重いだけでなく、評価時の必要kも増えるはずです。CPUプロトタイプとしては二重に厳しくなります。

9. いまのロードマップをどう直すべきか

現在のM7/M8はこう見えます。

M5 tiny1var成功
↓
M6 best-of-k確認
↓
M7 proto3varを一気に大きくする
↓
M8 ベンチマーク

これを以下に変えるのが良いです。

M5 tiny1var成功
↓
M6 best-of-k確認
↓
M7a 3変数・小語彙・軽量encoder
↓
M7b 3変数・L=32
↓
M7c 演算子追加
↓
M7d encoder高度化
↓
M7e 80k式
↓
M8 Nguyen / Feynman easy / gplearn比較

つまり、M7を単一マイルストーンではなく、スケールアップ実験群に分割するべきです。

10. 実務的な次の一手

今すぐやるなら、順番はこれです。

1. 現在のM7実行を停止する。
2. M7の設定・学習ループ・モデル定義を添付する。
3. M7-lite を作る。
4. 50 batchだけ走らせて、1 batchあたりの秒数を測る。
5. PMAあり/なし、n_points=32/64/128、d_model=64/128でミニ比較する。
6. 1 epoch 30分以内に収まる構成をM7正式版にする。
7. その構成でin-distribution 50〜100式を評価する。
8. Nguyenに進む。

現段階のプロジェクト評価は、研究テーマとしては有望、M6まではかなり良い、M7は設計上のスケールジャンプが大きすぎるので、実験計画を刻むべきです。