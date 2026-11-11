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
* **$S(t) \in \mathbb{R}^{d_m \times d_m}$**: $t$ anındaki dinamik ilişkisel bellek durumu (Memory state matrix).

Ağırlık Matrisleri ve Yanılgı Vektörleri (Öğrenilebilir Parametreler):
* **$W_k \in \mathbb{R}^{d_m \times d_e}$**, $b_k \in \mathbb{R}^{d_m}$: Anahtar (Key) projeksiyonu
* **$W_v \in \mathbb{R}^{d_m \times d_e}$**, $b_v \in \mathbb{R}^{d_m}$: Değer (Value) projeksiyonu
* **$W_q \in \mathbb{R}^{d_m \times d_e}$**, $b_q \in \mathbb{R}^{d_m}$: Sorgu (Query) projeksiyonu
* **$W_\eta \in \mathbb{R}^{1 \times d_e}$**, $b_\eta \in \mathbb{R}$: Güncelleme katsayısı (Update gate) projeksiyonu
* **$W_\gamma \in \mathbb{R}^{1 \times d_e}$**, $b_\gamma \in \mathbb{R}$: Koruma katsayısı (Retain gate) projeksiyonu

## 2. Başlangıç Koşulları

Her dizinin başlangıcında, $t=0$ için ilişkisel bellek matrisi boş (sıfır matrisi) olarak başlatılır:
$$S(0) = 0 \in \mathbb{R}^{d_m \times d_m}$$

## 3. Anahtar (K), Değer (V) ve Sorgu (Q) Hesaplamaları

Zaman adımı $t$ için girdi vektörü $x_t$'den hareketle ham anahtar, değer ve sorgu vektörleri doğrusal dönüşüm ile elde edilir:

* Ham Anahtar: $\tilde{k}(t) = W_k x_t + b_k$
* Ham Değer: $\tilde{v}(t) = W_v x_t + b_v$
* Ham Sorgu: $\tilde{q}(t) = W_q x_t + b_q$

Kararlılığı sağlamak için bu vektörler $L_2$ normu üzerinden normalize edilir:

* **Anahtar**: $k(t) = \frac{\tilde{k}(t)}{\|\tilde{k}(t)\|_2} \in \mathbb{R}^{d_m}$
* **Değer**: $v(t) = \frac{\tilde{v}(t)}{\|\tilde{v}(t)\|_2} \in \mathbb{R}^{d_m}$
* **Sorgu**: $q(t) = \frac{\tilde{q}(t)}{\|\tilde{q}(t)\|_2} \in \mathbb{R}^{d_m}$

## 4. Öğrenme ve Koruma Katsayılarının Hesaplanması

Modelin, eski bilgiyi koruma oranını belirleyen $\gamma(t)$ ve yeni bilgiyi kaydetme oranını belirleyen $\eta(t)$ katsayıları, sigmoid aktivasyon fonksiyonu $\sigma(\cdot)$ ile $[0, 1]$ aralığına sıkıştırılır:

* **Güncelleme Katsayısı (Update Gate)**:
  $$\eta(t) = \sigma(W_\eta x_t + b_\eta) \in (0, 1)$$

* **Koruma Katsayısı (Retain Gate)**:
  $$\gamma(t) = \sigma(W_\gamma x_t + b_\gamma) \in (0, 1)$$

## 5. Bellek Durumu Geçişi ve Güncelleme Denklemi

Belleğe bilgi ekleme ve mevcut bilgiyi revize etme işlemleri, mevcut bellek durumu $S(t)$'nin anahtar $k(t)$ ile çarpılması sonucu elde edilen tahmini (okunan) değer ile asıl değer $v(t)$ arasındaki fark üzerinden yapılır.

* Fark (Diff) vektörü:
  $$\Delta(t) = v(t) - S(t)k(t) \in \mathbb{R}^{d_m}$$

* Güncelleme Terimi (Update term): Dış çarpım kullanılarak hesaplanır:
  $$U(t) = \Delta(t) \otimes k(t) = \Delta(t) k(t)^\top \in \mathbb{R}^{d_m \times d_m}$$

* **Son Bellek Güncelleme Denklemi**:
  $$S(t+1) = \gamma(t)S(t) + \eta(t) U(t)$$

  *Tam açılımıyla:*
  $$S(t+1) = \gamma(t)S(t) + \eta(t)[v(t) - S(t)k(t)]k(t)^\top$$

## 6. Bellekten Okuma (Çıkarım) İşlemi

Her bir zaman adımında güncellenmiş bellek durumu $S(t+1)$, sorgu vektörü $q(t)$ kullanılarak okunur. Okunan değer $r(t)$ şu şekilde hesaplanır:

* **Okuma (Read) Denklemi**:
  $$r(t) = S(t+1) q(t) \in \mathbb{R}^{d_m}$$

Elde edilen $r(t)$ vektörü, asıl girdi $x_t$ ile birleştirilerek (concatenate) çıktı katmanlarına yönlendirilir.
