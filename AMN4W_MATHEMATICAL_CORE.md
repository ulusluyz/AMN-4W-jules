# AMN-4W Matematiksel Çekirdek

AMN-4W mimarisi, bilgiyi zaman adımına (t) bağlı olarak dinamik bir ilişkisel bellekte güncelleyen ve okuyan otoregresif bir mekanizmadır. Bu doküman, modelin temelini oluşturan işlemlerin saf matematiksel gösterimlerini içermektedir.

## 1. Değişkenler, Matrisler ve Boyutlar

Matematiksel işlemlerde kullanılan temel notasyonlar ve boyutlar aşağıdadır:

* **$b$**: Yığın boyutu (Batch size)
* **$L$**: Dizi uzunluğu (Sequence length)
* **$d_{e}$**: Gömme boyutu (Embedding dimension)
* **$d_{m}$**: Bellek boyutu (Memory dimension) - Anahtar ve değer vektörlerinin boyutu.

Girdi ve Bellek Durumu:
* **$x_t \in \mathbb{R}^{d_e}$**: $t$ anındaki girdi vektörü (Gömme ve pozisyonel kodlama toplamı).
* **$S_t \in \mathbb{R}^{d_m \times d_m}$**: $t$ anındaki dinamik ilişkisel bellek durumu.

Ağırlık Matrisleri ve Yanılgı Vektörleri (Öğrenilebilir Parametreler):
* **$W_k \in \mathbb{R}^{d_m \times d_e}$**, $b_k \in \mathbb{R}^{d_m}$: Anahtar (Key) projeksiyonu
* **$W_v \in \mathbb{R}^{d_m \times d_e}$**, $b_v \in \mathbb{R}^{d_m}$: Değer (Value) projeksiyonu
* **$W_q \in \mathbb{R}^{d_m \times d_e}$**, $b_q \in \mathbb{R}^{d_m}$: Sorgu (Query) projeksiyonu
* **$W_\eta \in \mathbb{R}^{1 \times d_e}$**, $b_\eta \in \mathbb{R}$: Güncelleme katsayısı projeksiyonu
* **$W_\gamma \in \mathbb{R}^{1 \times d_e}$**, $b_\gamma \in \mathbb{R}$: Koruma katsayısı projeksiyonu

## 2. Başlangıç Koşulları

Her dizinin başlangıcında, $t=0$ için ilişkisel bellek matrisi boş (sıfır matrisi) olarak başlatılır:
$$S_0 = 0 \in \mathbb{R}^{d_m \times d_m}$$

## 3. Anahtar (K), Değer (V) ve Sorgu (Q) Hesaplamaları

Zaman adımı $t$ için girdi vektörü $x_t$'den hareketle ham anahtar, değer ve sorgu vektörleri doğrusal dönüşüm ile elde edilir.

Sıfır bölme hatalarını (Zero-norm) engellemek için normalizasyona $\epsilon$ (küçük bir pozitif sabit, örn: $10^{-8}$) eklenir:

* **Anahtar**: $k_t = \frac{W_k x_t + b_k}{\|W_k x_t + b_k\|_2 + \epsilon} \in \mathbb{R}^{d_m}$
* **Değer**: $v_t = \frac{W_v x_t + b_v}{\|W_v x_t + b_v\|_2 + \epsilon} \in \mathbb{R}^{d_m}$
* **Sorgu**: $q_t = \frac{W_q x_t + b_q}{\|W_q x_t + b_q\|_2 + \epsilon} \in \mathbb{R}^{d_m}$

## 4. Öğrenme ve Koruma Katsayılarının Hesaplanması

Modelin genel koruma oranını belirleyen $\gamma_t$ ve yeni bilgiyi kaydetme oranını belirleyen $\eta_t$ katsayıları, sigmoid aktivasyon fonksiyonu $\sigma(\cdot)$ ile $[0, 1]$ aralığına sıkıştırılır:

* **Güncelleme Katsayısı (Update Gate)**: $\eta_t = \sigma(W_\eta x_t + b_\eta) \in (0, 1)$
* **Koruma Katsayısı (Retain Gate)**: $\gamma_t = \sigma(W_\gamma x_t + b_\gamma) \in (0, 1)$

## 5. Bellek Durumu Geçişi, Güncelleme ve Seçici Silme

Temel güncelleme denklemi şudur:
$$S_{t+1} = \gamma_t S_t + \eta_t [v_t - S_t k_t] k_t^\top$$

Bu formüldeki mekanizmaların karşılığı:
- **Belleğe Yazma ve Mevcut Bilgiyi Güncelleme**: Hata terimi $(v_t - S_t k_t)$, yeni değer ile belleğin mevcut anahtara verdiği tepki arasındaki farkı ölçer. $\eta_t$ ile ağırlıklandırılıp $k_t^\top$ ile çarpılarak belleğe yazılır.
- **Eski Bilgiyi Koruma**: $\gamma_t$ tüm matrisi etkileyen global bir bozunma katsayısıdır.
- **Seçici Bilgi Silme (Selective Deletion)**: $\gamma_t$ global olduğu için tek bir bilgiyi silmek $\gamma_t$ ile yapılmaz. Spesifik bir $k_t$ anahtarına ait bilgi silinmek istendiğinde, model hedeflenen bilginin değer vektörünü sıfıra yakınsar ($v_t \to 0$ veya $v_t \to v_{null}$). Bu durumda hata terimi $(0 - S_t k_t) = -S_t k_t$ olur. Böylece denklem $S_{t+1} = S_t - \eta_t (S_t k_t) k_t^\top$ şeklini alır ve yalnızca $k_t$ yönündeki bilgi bellekten seçici olarak silinir. Diğer ortogonal bilgiler bu işlemden etkilenmez.

## 6. Çok Aşamalı (İki Aşamalı) Çıkarım Mekanizması

İlişkisel çıkarım, belleğin önceki okuma çıktılarının ardışık adımlarda yeniden bellek sorgulamasına dönüştürülmesiyle elde edilir (Multi-hop reasoning).

İlk okuma $r_t^{(1)}$, $x_t$'den elde edilen sorgu ile yapılır:
$$r_t^{(1)} = S_{t+1} q_t$$

Eğer iki aşamalı (2-hop) çıkarım gerekiyorsa, model önceki okuma çıktısını yeni bir sorguya dönüştürür:
$$q_t^{(2)} = \frac{W_q^{(2)} r_t^{(1)} + b_q^{(2)}}{\|W_q^{(2)} r_t^{(1)} + b_q^{(2)}\|_2 + \epsilon}$$
$$r_t^{(2)} = S_{t+1} q_t^{(2)}$$

Bu ardışık (iteratif) bellek okumaları $N$ adım boyunca tekrarlanarak zincirleme (ilişkisel) çıkarım yeteneğini (örneğin A -> B -> C mantığı) matematiksel olarak ağa dahil eder.

## 7. Otoregresif Çıktı Katmanı

Entegre okuma $r_t$ (veya çok aşamalı çıkarım için $r_t^{(N)}$), girdi vektörü $x_t$ ile birleştirilip lineer projeksiyonlar ve bir aktivasyon (örneğin ReLU) ile geçirilerek kelime (token) olasılıklarını üretir.

* $h_t = \text{ReLU}(W_{h} [x_t \oplus r_t] + b_{h})$
* Çıktı Olasılıkları (Logits): $P(y_{t+1} | x_1 ... x_t) = \text{Softmax}(W_{out} h_t + b_{out})$

Burada $\oplus$ operatörü vektör birleştirme (concatenation) işlemini temsil eder.
