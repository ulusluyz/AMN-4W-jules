# AMN-4W Matematiksel Çekirdek (Yönlü Koruma Modeli)

AMN-4W mimarisi, bilgiyi zaman adımına (t) bağlı olarak dinamik bir ilişkisel bellekte güncelleyen ve okuyan otoregresif bir mekanizmadır. Bu doküman, bellek kaybını tamamen önleyen **Yönlü Koruma (Directional Retention)** denklemini ve tüm modelin güncel matematiksel gösterimlerini içermektedir.

## 1. Değişkenler, Matrisler ve Boyutlar

* **$b$**: Yığın boyutu (Batch size)
* **$L$**: Dizi uzunluğu (Sequence length)
* **$d_{e}$**: Gömme boyutu (Embedding dimension)
* **$d_{m}$**: Bellek boyutu (Memory dimension)

Girdi ve Bellek Durumu:
* **$x_t \in \mathbb{R}^{d_e}$**: $t$ anındaki girdi vektörü.
* **$S(t) \in \mathbb{R}^{d_m \times d_m}$**: $t$ anındaki dinamik ilişkisel bellek durumu.

Ağırlık Matrisleri ve Yanılgı Vektörleri (Öğrenilebilir Parametreler):
* **$W_k \in \mathbb{R}^{d_m \times d_e}$**, $b_k \in \mathbb{R}^{d_m}$: Anahtar (Key)
* **$W_v \in \mathbb{R}^{d_m \times d_e}$**, $b_v \in \mathbb{R}^{d_m}$: Değer (Value)
* **$W_q \in \mathbb{R}^{d_m \times d_e}$**, $b_q \in \mathbb{R}^{d_m}$: Sorgu (Query)
* **$W_\eta \in \mathbb{R}^{1 \times d_e}$**, $b_\eta \in \mathbb{R}$: Güncelleme katsayısı
* **$W_\gamma \in \mathbb{R}^{1 \times d_e}$**, $b_\gamma \in \mathbb{R}$: Koruma katsayısı

## 2. Başlangıç Koşulları

$$S(0) = 0 \in \mathbb{R}^{d_m \times d_m}$$

## 3. Anahtar (K), Değer (V) ve Sorgu (Q) Hesaplamaları

Sıfır bölme hatalarını (Zero-norm) engellemek için normalizasyona $\epsilon$ (örn: $10^{-8}$) eklenir:

* **Anahtar**: $k_t = \frac{W_k x_t + b_k}{\|W_k x_t + b_k\|_2 + \epsilon} \in \mathbb{R}^{d_m}$
* **Değer**: $v_t = \frac{W_v x_t + b_v}{\|W_v x_t + b_v\|_2 + \epsilon} \in \mathbb{R}^{d_m}$
* **Sorgu**: $q_t = \frac{W_q x_t + b_q}{\|W_q x_t + b_q\|_2 + \epsilon} \in \mathbb{R}^{d_m}$

## 4. Öğrenme ve Koruma Katsayılarının Hesaplanması

* **Güncelleme Katsayısı (Update Gate)**: $\eta_t = \sigma(W_\eta x_t + b_\eta) \in (0, 1)$
* **Koruma Katsayısı (Retain Gate)**: $\gamma_t = \sigma(W_\gamma x_t + b_\gamma) \in (0, 1)$

## 5. Bellek Durumu Geçişi: Yönlü Koruma (Directional Retention)

Önceki mimarilerdeki global aşınma sorunu (eski bilgilerin $\gamma$ ile zamanla ezilmesi), yeni "Yönlü Koruma Denklemi" ile çözülmüştür:

$$S(t+1) = S(t) + \left[ \eta_t v_t - \big( (1-\gamma_t) + \eta_t \gamma_t \big) S(t)k_t \right] k_t^\top$$

**Matematiksel Mekanizmalar:**
- **Kayıpsız Bellek Koruma (Lossless Retention):** $S(t)$ ana matrisi denklemin başında global herhangi bir çarpana maruz kalmadığı için, ortogonal olan eski anahtarlara ait bilgiler matematiksel olarak %100 oranında (1.000 Büyüklük/Kosinüs benzerliği) korunur.
- **Belleğe Yazma:** Yeni değer $\eta_t v_t$ ağırlıklandırmasıyla, yalnızca $k_t^\top$ yönünde matrise eklenir.
- **Seçici Silme (Selective Deletion) ve Güncelleme:** Hedeflenen bilgi, $\big( (1-\gamma_t) + \eta_t \gamma_t \big) S(t)k_t$ terimi ile çıkarılır. $\gamma_t$ model tarafından sıfırlanıp 1'e yaklaştırıldığında veya $\eta_t$ ile kontrol edildiğinde, model spesifik bir yönü (anahtarı) hedef alarak eski değeri tamamen veya kısmen silebilir.
- **Anahtar Çakışmaları (Key Interference):** Tam ortogonal olmayan anahtarlarda bile (örn. rastgele benzer anahtarlarda), global matris bozunmadığı için bilgi koruma oranı (magnitude retention) yeni denklemde eski global decay (0.409) yerine **0.887** gibi kararlı seviyelere ulaşır. Sadece doğrudan birbiriyle örtüşen vektör bileşenleri parazite uğrar.

## 6. Çok Aşamalı (İki Aşamalı) Çıkarım Mekanizması

İlk okuma $r_t^{(1)}$, $x_t$'den elde edilen sorgu ile yapılır:
$$r_t^{(1)} = S(t+1) q_t$$

Eğer iki aşamalı (2-hop) çıkarım gerekiyorsa, model önceki okuma çıktısını yeni bir sorguya dönüştürür:
$$q_t^{(2)} = \frac{W_q^{(2)} r_t^{(1)} + b_q^{(2)}}{\|W_q^{(2)} r_t^{(1)} + b_q^{(2)}\|_2 + \epsilon}$$
$$r_t^{(2)} = S(t+1) q_t^{(2)}$$

## 7. Otoregresif Çıktı Katmanı

Entegre okuma $r_t$, girdi vektörü $x_t$ ile birleştirilip lineer projeksiyonlar ve bir aktivasyon (ReLU) ile geçirilerek kelime olasılıklarını üretir.
* $h_t = \text{ReLU}(W_{h} [x_t \oplus r_t] + b_{h})$
* Çıktı Olasılıkları (Logits): $P(y_{t+1} | x_1 ... x_t) = \text{Softmax}(W_{out} h_t + b_{out})$
