from flask import Flask, render_template
from flask import request
from translate import Translator_Inference
from docx import Document
from io import StringIO

app = Flask(__name__, template_folder='templates')

@app.route("/", methods=["POST", "GET"])
def index():
    text = ""

    if request.method=="POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename=="":
            return redirect(request.url)
        
        if file:
            source = request.form.get(srcLang)
            target = request.form.get('targetLang')
            asr = translator_inference(file, source, target)
            document = asr.run_translate()
            f = BytesIO()
            document.save(f)
            f.seek(0)
            new_file_name = file.filename + "_translated"
    return send_file(f,
                     as_attachment=True,
                     attachment_filename=new_file_name)


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
#     target = 
#             translated_text = asr.get_languages("english","spanish")

#     return render_template("upload.html", text = translated_text)