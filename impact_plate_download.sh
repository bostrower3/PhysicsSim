URL="https://figshare.com/ndownloader/articles/24198399?private_link=a2c4abb9872b1dae3286"
OUTPUT_DIR="${1:-impact_plate}"

mkdir -p "$OUTPUT_DIR"

for i in {1..30}; do
    echo "Attempt $i..."
    rm -f data.zip

    wget -O data.zip "$URL" || true

    if [ -s data.zip ] && unzip -t data.zip >/dev/null 2>&1; then
        echo "Valid zip downloaded"
        break
    fi

    echo "Not ready yet; waiting 5 seconds..."
    sleep 5
done

ls -lh data.zip
unzip data.zip -d "$OUTPUT_DIR"
rm data.zip