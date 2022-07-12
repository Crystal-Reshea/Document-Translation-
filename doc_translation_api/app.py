from flask import Flask, render_template
from flask import request, send_file
from fixed_translate import Translator_Inference
from docx import Document
from io import StringIO
from io import BytesIO

app = Flask(__name__, template_folder='templates')
global fn
fn = "No jobs completed"
@app.route("/", methods=["POST", "GET"])
def index():
    text = "" 
    fn = ""
    if request.method=="POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename=="":
            return redirect(request.url)
        
        if file:
            source = request.form.get('srcLang')
            target = request.form.get('targetLang')
            asr = Translator_Inference(file, source, target)
            document = asr.run_translate()
            fn = 'translated_doc'
            document.save(fn)
            # f = BytesIO()
            # document.save(f)
            # f.seek(0)
    return render_template("upload.html", text = str(fn))
    
@app.route('/return-files/')
def return_files_tut():
    try:
        return send_file('/Users/reshea/Documents/Work/doc_translation/doc_translation_api/translated_doc', attachment_filename='translated_doc.docx')
    except Exception as e:
        return str(e)


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
#     target = 
#             translated_text = asr.get_languages("english","spanish")

#     return render_template("upload.html", text = translated_text)