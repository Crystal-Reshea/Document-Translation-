import 


class translator_inference():
     def __init__(self, file): 
            self.file_type = self.get_type(file)
            self.translate_languages = {
                'english-spanish':"Helsinki-NLP/opus-mt-en-es",
                'spanish-english': "Helsinki-NLP/opus-mt-es-en",
                'french-english': "Helsinki-NLP/opus-mt-fr-en",
                'english-french':"Helsinki-NLP/opus-mt-en-fr",
                'english-japanese':"Helsinki-NLP/opus-tatoeba-en-ja",
                'japanese-english':"Helsinki-NLP/opus-mt-ja-en"
            }
            self.audio_reader = 'patrickvonplaten/wav2vec2-base-100h-with-lm'
            self.file = file
   
    def get_type(self, file):
        if file.endswith('docx'): 
            return 'docx'
        elif file.endswith(('m4a', 'wav','flac', 'mp3', 'wma', 'aac')):
            return 'audio'
    
    def get_model(self, source, target):
        search = source + '-' + target
        if search in self.translate_languages: 
            self.pipeline = pipeline('translation', model = self.translate_languages[search])
            return self.pipeline
        else:
            print("Source and Target languages not yet available.")
    
    def get_text(self, file):
        if self.file_type is 'audio':
        elif self.file_type is 'docx':
            self.doc = docx.Document(self.file)
        else:
            print("please insert audio or docx files")
    
    def getText(self, doc):
        fullText = []
        # Reading in paragraphs
        for para in doc.paragraphs:
            fullText.append(para.text)
        # Reading in text from tables
        for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            fullText.append(paragraph.text)
        return '\n'.join(fullText)
    
    def translate_paragraphs(paragraphs):
        lt = LineTokenizer()

        # To add more languages
        if torch.cuda.is_available():  
          dev = "cuda"
        else:  
          dev = "cpu" 
        device = torch.device(dev)

        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        model.to(device)

        # Translating sentences
        # Could choose batch size dynamically based on the length of the sentences.
        # Ideally no sentence/batch will be greater than 209 words
        batch_size = 8
        keys = []
        translated_paragraphs = []
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            batches = math.ceil(len(sentences) / batch_size)     
            translated = []
            for i in range(batches):
                # selecting the sentences to batch
                sent_batch = sentences[i*batch_size:(i+1)*batch_size]
                keys.extend(sent_batch)
                model_inputs = tokenizer(sent_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    translated_batch = model.generate(**model_inputs)
                translated += translated_batch
            translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            translated_paragraphs += [" ".join(translated)]

        translated_text = "\n".join(translated_paragraphs)
        return translated_paragraphs, keys
    
    def replace_text_in_paragraph(paragraph, key, value):
        change_tracker = False
        if key in paragraph.text:
            inline = paragraph.runs
            for item in inline:
                if key in item.text:
                    item.text = item.text.replace(key, value)
                    change_tracker = True
            if change_tracker == False:
                paragraph.text = paragraph.text.replace(key, value)
   
    def docx_replace(doc_obj, pairs, output_file_path):
        for key, value in pairs.items():
            for paragraph in doc_obj.paragraphs:
                replace_text_in_paragraph(paragraph, key, value)
            for table in doc_obj.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            replace_text_in_paragraph(paragraph, key, value)
        return doc_obj.save(output_file_path)
        print("Done")
    
    def translate_audio(file):
    
    def process_docx_text(docx_text): 
        
    def translate_text(text):
    