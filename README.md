# 🎵 Music Studio AI (Rust + Local ONNX)

Ứng dụng xử lý âm thanh chuyên nghiệp viết bằng **Rust**, hỗ trợ:
1. **Nâng hạ Tone (Pitch Shifting) & Chỉnh tốc độ (Time Stretching)** thời gian thực có bảo toàn âm sắc (**Formant Preservation**) không gây méo tiếng.
2. **Tách Nhạc Nền (Instrumental), Vocal Chính (Lead Vocal) và Giọng Bè (Backing Vocal / Choir)** bằng các mô hình AI mã nguồn mở miễn phí chạy 100% Local (ONNX Runtime).
3. **Giao diện chuẩn Phòng thu (Dark Mode Audio Studio)** thiết kế với [`nextlevelbuilder/ui-ux-pro-max-skill`](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill).

👉 **Xem toàn bộ Bản Kế Hoạch Kỹ Thuật Chi Tiết tại:** [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md)

---

## 🧠 Danh Mục Model AI Chạy Local (Miễn Phí 100%)

| Nhiệm vụ | Model (`.onnx`) | Kiến trúc | Dung lượng |
| :--- | :--- | :--- | :--- |
| **Tách Nhạc Nền & Tổng Vocal (Nhanh)** | `UVR-MDX-NET-Inst_HQ_4.onnx` | MDX-Net | ~56 MB |
| **Tách Nhạc Nền & Tổng Vocal (Siêu Nét)** | `mel_band_roformer_kim_ft_unwa.onnx` | Mel-Band RoFormer | ~228 MB |
| **Tách Vocal Chính & Giọng Bè** | `UVR_MDXNET_KARA_2.onnx` / `mel_band_roformer_karaoke_aufr33_viperx.onnx` | MDX-Net / RoFormer Karaoke | ~55 MB / ~228 MB |
| **Tách Đuôi Vang (De-Reverb)** | `Reverb_HQ_By_FoxJoy.onnx` | MDX-Net De-Reverb | ~67 MB |
| **Nâng Hạ Tone AI (Vocal Render)** | `rmvpe.onnx` + `nsf_hifigan.onnx` + `WORLD Vocoder` | Neural Source-Filter | ~90 MB |

---

## 🛡️ Tối Ưu Hóa Chống Lag Trên Máy Công Ty (AhnLab Endpoint Security)

Dự án được cấu hình sẵn trong [`.cargo/config.toml`](./.cargo/config.toml) để tránh tình trạng trình bảo mật **AhnLab** quét hàng loạt file `.exe` sinh ra cùng lúc khi build Rust:
- **Kiến trúc Single-Binary (1 file `.exe` duy nhất):** Không dùng sidecar ngoài (`ffmpeg.exe`, `python.exe`), không chia nhỏ nhiều crate tạo nhiều `build-script-build.exe`.
- **100% Pure Rust Audio & FFT (`symphonia`, `realfft`, `rubato`):** Không cần trình biên dịch C/C++ sinh file thực thi tạm lúc build.
- **ONNX Runtime Dynamic (`load-dynamic`):** Nạp trực tiếp `onnxruntime.dll` có chữ ký số chính thức của Microsoft.
- **Giới hạn luồng biên dịch (`jobs = 2`) & Tắt Debug Symbol thừa (`strip = "debuginfo"`):** Giảm dung lượng file `.exe` xuống ~10 lần và ngăn chặn bão tiến trình khi build.
