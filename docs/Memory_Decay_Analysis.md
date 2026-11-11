# Bellek Kaybı (Memory Decay) ve Çözüm Analizi

## Tespit Edilen Sorun
Önceki formülde yer alan `S(t+1) = \gamma(t) S(t) + ...` denklemindeki $\gamma$ (koruma) katsayısı, matrisi her işlemde skaler olarak küçültmektedir. Yeni ve alakasız bilgiler öğrenilirken bile $\gamma$ 1.0'ın altında kaldığı için (örn: 0.95), eski bilgiler global bir şekilde zayıflamaya başlar. Rastgele anahtarlarla yapılan 20 işlemlik bir deneyde, hedef bilginin ilk okuma büyüklüğüne (magnitude) oranı **0.409**'a kadar düşmüştür (belirtilen 0.410 sorununu tam olarak doğrulamaktadır). Anahtarlar tamamen ortogonal olsa bile sinyal büyüklüğü %64 oranında kaybedilmektedir.

## Geliştirilen Matematiksel Çözüm
Global bir bozunma (decay) yerine, silme ve güncelleme işlemlerinin **sadece hedeflenen anahtar (k_t) yönünde** yapılmasını sağlayan "Yönlü Unutma Kapısı" (Directional Forget Gate) tasarlanmıştır:

**Yeni Denklem:**
$$S_{t+1} = S_t + [ \eta_t v_t - \gamma_t S_t k_t ] k_t^\top$$

Bu denklemde $S_t$'nin geri kalanı korunurken, yalnızca $k_t$ doğrultusundaki bilgi $\gamma_t$ oranında zayıflatılır ve yeni değer $\eta_t v_t$ eklenir.

## Deney Sonuçları (20 İşlem Sonrası Bellek Koruma)

1. **Eski Formül (Global Gamma):**
   * Rastgele Anahtarlar: Sinyal Büyüklüğü Korunumu: **0.409**
   * Tam Ortogonal Anahtarlar: Sinyal Büyüklüğü Korunumu: **0.358** (Sinyal eziliyor)

2. **Yeni Formül (Directional Forget Gate):**
   * Rastgele Anahtarlar: Sinyal Büyüklüğü Korunumu: **0.893** (Çok daha kararlı)
   * Tam Ortogonal Anahtarlar: Sinyal Büyüklüğü Korunumu: **1.000** (Kayıpsız tam koruma)

Bu çözüm sayesinde bellek kapasitesi, alakasız bilgi aktarımlarından kaynaklı silinmelere karşı bağışık hale getirilmiş ve yalnızca hedeflenen güncelleme/silme işlemleri güvenle yapılabilir olmuştur. Anahtar çakışması (key interference) durumunda minimal kayıp yaşansa da, sinyalin ezilme problemi çözülmüştür.
