# RVC-realtime-for-xpu

RVCのxpu向け修正

</br>

## これは何

https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

これのリアルタイム推論を、Intel Arc向けにちょっと改造したフォーク版。

深夜テンションで作った分どこかおかしいかも。

</br>

## 使い方

### uvが入ってること前提

(uv入れるだけならそんなむずくないはず)

</br>

Releasesから7zファイルをダウンロード、展開。

その後setup.batを実行した後にgo-realtime_gui.batを実行するだけ。GUIの使い方は本家参照。

あと、トレーニングとかの部分は確認してないので削除してます。(面倒くさかったともいう)

</br>

これも参考に :

https://www.youtube.com/watch?v=Pjf4gTvB4QU

</br>

## 確認済みのバグ

~~声の太さを0にすると、少し機械音が混じる。~~

~~0以外だと発生しないので調べればわかるだろうが、そこまでの気力がないから君たちで頑張って。~~

修正済み

</br>

## 修正箇所

chatGPTにまとめてもらった

</br>

1. infer/module/models.py

RVCのDecoder内にある noise_convs[0] の処理をXPU向けに変更しました。

元：

```python
x_source = noise_convs(har_source)
```

変更後：

```python
if i == 0:
    x_source = F.conv1d(
        F.pad(har_source, (20, 20)),
        noise_convs.weight,
        noise_convs.bias,
        stride=40,
        padding=0,
        dilation=1,
        groups=1,
    )
else:
    x_source = noise_convs(har_source)
```

目的は、Intel XPU上で Conv1d(..., kernel_size=80, stride=40, padding=20) をそのまま実行した際に発生していたかすかす・音質異常を回避することです。

</br>

2. PyTorchをXPU版へ変更

使用するPyTorchを、

```text
torch==2.7.1+xpu
torchaudio==2.7.1+xpu
```

にしました。

取得元：

https://download.pytorch.org/whl/xpu

</br>

3. XPU用の依存関係ファイルを作成

requirments_xpu_py312.txt

を作成し、Python 3.12 + Intel XPU環境で必要な依存パッケージをまとめました。

</br>

4. setup.bat でXPU環境を自動構築

setup.bat で、

```text
uv
↓
Python 3.12.13
↓
.venv
↓
torch 2.7.1+xpu
torchaudio 2.7.1+xpu
↓
XPU用依存パッケージ
```

を自動構築するようにしました。

PyTorch XPUの取得では、

```text
--index-url https://download.pytorch.org/whl/xpu
--extra-index-url https://pypi.org/simple
--index-strategy unsafe-best-match
```

を使用しています。

</br>

5. CUDA前提だった箇所をXPU環境で動作確認

実際の環境で、

```text
CUDA利用可能: False
```

でもRVCが動作し、

```text
torch.xpu
```

を使用する環境でリアルタイム推論できることを確認しました。

つまり、XPU化そのものに関係する変更は「PyTorch XPU化」「XPU環境構築」「Conv1d padding=20 の回避」の3系統です。

</br>

6. 追記

### XPU向け機械音対策

### 修正箇所

`infer/module/models.py`

`GeneratorNSF.forward()` 内の `har_source` 処理を修正しました。

#### 修正前

```python
har_source, noi_source, uv = self.m_source(f0, self.upp)
har_source = har_source.transpose(1, 2)
```

#### 修正後

`n_res` による長さ調整処理の後に、以下を追加しました。

```python
har_source = har_source.clone(memory_format=torch.contiguous_format)
```

最終的には以下の位置になります。

```python
har_source, noi_source, uv = self.m_source(f0, self.upp)
har_source = har_source.transpose(1, 2)

if n_res is not None:
    n = int(n_res.item()) if isinstance(n_res, torch.Tensor) else int(n_res)

    if n * self.upp != har_source.shape[-1]:
        har_source = F.interpolate(
            har_source,
            size=n * self.upp,
            mode="linear",
        )

    if n != x.shape[-1]:
        x = F.interpolate(
            x,
            size=n,
            mode="linear",
        )

har_source = har_source.clone(memory_format=torch.contiguous_format)

x = self.conv_pre(x)
```

### 修正の目的

`har_source` は `transpose(1, 2)` を通ることで、メモリ上の配置（stride）が変化します。

Intel XPU環境では、この状態の `har_source` を後段の処理へ渡した場合、

- 声の太さが `0`
- 声の太さが `0` 未満

のときに機械音が混じる問題が発生しました。

そこで、

```python
har_source = har_source.clone(memory_format=torch.contiguous_format)
```

によって、`har_source` を連続したメモリ配置のテンソルとして作り直してから後段へ渡すようにしました。

### 切り分け結果

以下の変更では問題は解消しませんでした。

```text
x.contiguous()
F.interpolate() のCPU化
ups[0] のCPU化
noise_convs[0] のCPU化
resblocks のCPU化
m_source のCPU化
F0補正の無効化
cache_pitchf の変更
後段Resampleの変更
```

一方、

```python
har_source = har_source.clone(memory_format=torch.contiguous_format)
```

を追加したところ、**声の太さが0および0未満の場合に発生していた機械音が解消**しました。

また、Decoder全体をCPUで実行した場合にも機械音が消えることを確認しています。

### まとめ

今回の修正は、RVCの音声処理そのものを変更するものではありません。

`transpose()` 後の `har_source` を連続メモリ配置へ変換し、**Intel XPU上で後段の処理が不正なメモリレイアウトを扱うことによる音質異常を回避するための修正**です。

追加した変更は以下の1行です。

```python
har_source = har_source.clone(memory_format=torch.contiguous_format)
```

</br>

大体あってそう。多分

</br>

## ライセンス

本家RVC様の規約にすべて従います。

また、ライセンス的にまずいところがあれば連絡してください。確認次第すぐに対応いたします。
