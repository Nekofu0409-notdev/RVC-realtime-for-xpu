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

声の太さを0にすると、少し機械音が混じる。

0以外だと発生しないので調べればわかるだろうが、そこまでの気力がないから君たちで頑張って。

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

大体あってそう。多分

</br>

## ライセンス

本家RVC様の規約にすべて従います。

また、ライセンス的にまずいところがあれば連絡してください。確認次第すぐに対応いたします。
