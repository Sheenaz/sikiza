#!/usr/bin/env sh
# Download the trained serving bundle (AfroXLM-R + LoRA, ONNX int8, 279 MB; v0.2.0 = trained on real Zindi data) from the GitHub
# release into artifacts/model/. It is too large for a git repository (GitHub's limit is 100 MB).
# To produce it yourself instead: make demo  (or the real-data steps in README.md).
set -eu
VERSION="${1:-v0.2.0}"
URL="https://github.com/Sheenaz/sikiza/releases/download/${VERSION}/sikiza-model-${VERSION}.tar.gz"
case "$VERSION" in
  v0.2.0) SHA256="70dd4a32f45bdf503b9ad9ea2610060ffde49ba4f9a292bbf8b1649e5672535f" ;;  # real Zindi data
  v0.1.0) SHA256="f87a3ecd2469e27fca4b78c590d89a052bb7efa2fcf5a408a7f0b9ba0f329518" ;;  # synthetic demo data
  *) SHA256="" ;;
esac

cd "$(dirname "$0")/.."
mkdir -p artifacts
tmp="artifacts/.model-${VERSION}.tar.gz"
echo "Downloading ${URL}"
curl -fL --retry 3 -o "$tmp" "$URL"

if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$tmp" | cut -d' ' -f1)
else got=$(shasum -a 256 "$tmp" | cut -d' ' -f1); fi
if [ -n "$SHA256" ] && [ "$got" != "$SHA256" ]; then
  echo "Checksum mismatch: expected $SHA256, got $got" >&2; rm -f "$tmp"; exit 1
fi

rm -rf artifacts/model
tar -xzf "$tmp" -C artifacts
rm -f "$tmp"
echo "Model bundle ready in artifacts/model/:"
ls -l artifacts/model
