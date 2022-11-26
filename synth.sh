
        for emo in "Happy" "Sad" "Angry" "Neutral"; do
python3 synthesize_batch.py \
	--source qual_sentences.txt \
	--speaker_id "0019" \
	--emotion_id $emo \
	--arousal '-' \
	--valence '-' \
	--restore_step 80000 \
	--mode single \
	-p config/ESD/preprocess.yaml \
	-m config/ESD/model.yaml \
	-t config/ESD/train.yaml
done

