# taiwan-asr-eval

這是一組小工具，用公開語料比較開源語音辨識模型在台語、客語和東南亞語言上的表現。它不追求排行榜，目的是回答一個實際問題：一段家庭群組裡的語音訊息，現成的模型能不能聽懂，聽懂之後能不能拿去翻譯。

A small harness for comparing open speech-recognition models on Taiwanese Hokkien, Hakka and Southeast Asian languages using public test sets.

## 支援的測試集

| 名稱 | 語言 | 內容 | 來源 | 授權 |
|---|---|---|---|---|
| `cv_nan` | 台語 | Common Voice nan-tw 測試句，多為單詞、地名、俗語 | [Taiwan-Tongues-ASR-CE-dataset-hokkien](https://huggingface.co/datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien) test split | 見資料集卡 |
| `tt_hak` | 客語 | 合成語音，標註為華語 | [Taiwan-Tongues-ASR-CE-dataset-hakka](https://huggingface.co/datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hakka) test split | 見資料集卡 |
| `moe_hak` | 客語（四縣） | 教育部客語辭典例句錄音 | [ouob/hakkadict_210430_sixian](https://huggingface.co/datasets/ouob/hakkadict_210430_sixian) | 資料集卡未標示，使用前請自行確認 |
| `fleurs:<lang>` | 越南語 `vi_vn`、印尼語 `id_id`、泰語 `th_th`、菲律賓語 `fil_ph` 等 | FLEURS dev split 朗讀語音 | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | CC BY 4.0 |

本 repo 不附任何音檔，`fetch_data.py` 會從原始來源下載到 `data/`。

## 使用方式

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 下載測試集（不帶參數就全部下載，約 1.5 GB）
.venv/bin/python fetch_data.py cv_nan moe_hak fleurs:vi_vn

# 下載模型（CTranslate2 / faster-whisper 格式）
.venv/bin/hf download adi-gov-tw/Taiwan-Tongues-ASR-CE-v1.0 --local-dir models/tt
.venv/bin/hf download Systran/faster-whisper-large-v3 --local-dir models/wv3

# 評測：模型目錄、測試集、Whisper 語言代碼、句數、逐句結果輸出
.venv/bin/python asr_eval.py models/tt cv_nan zh 60 out/tt_cv_nan.jsonl
.venv/bin/python asr_eval.py models/wv3 fleurs:vi_vn vi 12 out/wv3_vi.jsonl
```

只有 Transformers 格式的模型（例如 [Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26)），要先轉成 CTranslate2 格式：

```bash
.venv/bin/pip install transformers torch
.venv/bin/ct2-transformers-converter --model MediaTek-Research/Breeze-ASR-26 \
  --output_dir models/breeze --quantization int8 \
  --copy_files tokenizer.json preprocessor_config.json
```

每次評測會印出一行摘要：句數、錯誤率和即時率（RTF，處理時間 ÷ 音檔長度，小於 1 代表比即時快）。
- 漢字語料和泰語計算字錯率（CER），其他語言計算詞錯率（WER）。
- 漢字比對前會先轉成繁體，並去掉標點和括號內的羅馬字。

## 2026-10 前測結果

在 Apple M4（10 核、CPU int8）上跑的結果：

| 語言 | 測試集 | 句數 | 模型 | 錯誤率 | RTF |
|---|---|---|---|---|---|
| 台語 | `cv_nan` | 60 | Breeze-ASR-26¹ | CER 75% | 4.4 |
| 台語 | `cv_nan` | 60 | Taiwan-Tongues-ASR-CE v1.0 | CER 95% | 2.8 |
| 客語 | `moe_hak` | 40 | Taiwan-Tongues-ASR-CE v1.0 | CER 54% | 2.5 |
| 客語 | `moe_hak` | 20 | Whisper large-v3 | CER 82% | 2.1 |
| 客語 | `tt_hak` | 20 | Taiwan-Tongues-ASR-CE v1.0 | CER 35% | 1.6 |
| 越南語 | `fleurs:vi_vn` | 12 | Whisper large-v3 | WER 13% | 1.5 |
| 印尼語 | `fleurs:id_id` | 12 | Whisper large-v3 | WER 4% | 1.3 |
| 泰語 | `fleurs:th_th` | 12 | Whisper large-v3 | CER 3% | 7.1 |
| 菲律賓語 | `fleurs:fil_ph` | 12 | Whisper large-v3 | WER 4% | 1.1 |

¹ 這次用的是社群轉好的 int8 CTranslate2 版本，不是自己從官方權重轉的。

**解讀時請注意：**
- **台語的字錯率會高估錯誤。** `cv_nan` 的標準答案是台語漢字（例如「我沒咧驚」），但這兩個模型輸出的都是華語漢字（例如「我沒在怕」）。意思對了，字錯率還是會算錯，所以這個數字只適合拿來比較模型之間的相對高低。
- **測試語料不是日常對話。** 台語、客語的語料多半是單詞、地名和俗語，比一般對話難；東南亞語言則是朗讀新聞句。真實的語音訊息會有背景噪音、語言夾雜和口語用字，表現可能更差。
- **句數少，數字只能看大方向。**

## 授權

程式碼採用 BSD 3-Clause 授權。測試集與模型各自依其原始授權。
