from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

text = "God protect me from short sightedness"

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")


# translate Chinese to English
tokenizer.src_lang = "en"
encoded_zh = tokenizer(text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("so"))
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(translation)