mkdir -p /Results
chmod +x /Results
python3 -m --model_path models/saved_model01.pth --entry_path /images/*1.jpg --store_path ./Snapshots/segmentations