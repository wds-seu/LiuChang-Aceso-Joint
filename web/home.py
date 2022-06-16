from flask import Flask, request, render_template, send_file
import json
import os
from pdf2txt.extractFullText import pdf2text
from pdf2txt.preprocess import preprosess_seme, preprosess_joint
from SEME import load_se_model, extract_medical_summary_1, seme_result_collation
from Joint import load_je_model, extract_medical_triples_1, joint_result_collation

# from web.web_util import save_input_sens, predict_and_show, predict_and_show_test

app = Flask(__name__)

global seme_model
global joint_model

@app.route('/home')
def ceg_home():
    return render_template('ceg_home.html')


def save_str_show_on_web(str_show_on_web, file_path):
    file_out = open(file_path, 'w')
    for type in ['p', 'i', 'o']:
        file_out.write("%s:\n" % type)
        dict = str_show_on_web[type]
        for token, label, sen in zip(dict['str'], dict['tag'], dict['sen']):
            file_out.write("%s %s[%s]\n" % (token, label, sen))
        file_out.write("\n")
    file_out.close()


@app.route('/extract_summary', methods=['POST', 'GET'])
def extract_summary():
    result = {"code": 0, "msg": "success", "result": ""}
    # SEME模型预测
    print("extract medical summary")
    seme_file = request.values.get('seme_path')
    result_path = request.values.get('save_path')
    seme_result = extract_medical_summary_1(seme_file, result_path, seme_model)
    print('extract summary success')
    # 处理预测结果
    result['result'], result['result_collection'] = seme_result_collation(seme_result, result_path)
    result['seme_result'] = seme_result
    return json.dumps(result)


@app.route('/extract_medical_evidence', methods=['POST', 'GET'])
def extract_medical_evidence():
    result = {"code": 0, "msg": "success", "result": ""}
    seme_result_file = request.values.get('seme_result_path')
    result_path = request.values.get('save_path')
    filename = request.values.get('filename')
    # 数据预处理
    joint_file = preprosess_joint(seme_result_file, filename, result_path)
    # Joint模型预测
    print("extract evidence")
    joint_results = extract_medical_triples_1(joint_file, result_path, joint_model)
    print('extract evidence success')
    # 处理预测结果
    result['result'], result['result_collection'] = joint_result_collation(joint_results, result_path)
    return json.dumps(result)


@app.route('/upload_medical', methods=['POST', 'GET'])
def upload_medical():
    file_dir = "input_of_web"  # 用户上传的文件存储路径
    file = request.files['medical-input']
    filename = file.filename[0:len(file.filename) - 4]
    save_dir = os.path.join(file_dir, filename)  # 存放相关的内容
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_file_name = os.path.join(save_dir, file.filename)
    file.save(save_file_name)

    # pdf转txt
    pdf2text(save_file_name, file)
    text = os.path.join(save_dir, file.filename[0:len(file.filename) - 4] + '.txt')
    print(text)

    # 写txt
    upload_text = ""
    if file and file.filename.endswith(".pdf"):
        # save_file_name = os.path.join(save_dir, "upload_%s" % file.filename)
        # file.save(save_file_name)
        with open(text, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                upload_text += line.strip()
                upload_text += " "

    # 预处理txt
    seme_file = preprosess_seme(text, filename, save_dir)
    # 结束
    print("uploaded file and preprocessed success")
    return json.dumps({"seme_path": seme_file, "save_path": save_dir, 'text': upload_text, 'filename':filename})


@app.route('/download_evidence', methods=['POST', 'GET'])
def download_evidence():
    file_name = request.values.get('filename')
    print(file_name)
    if not os.path.exists(file_name):
        return "error"
    return send_file(file_name, mimetype='text/csv/json', attachment_filename=file_name, as_attachment=True)

if __name__ == "__main__":
    seme_model = load_se_model()
    joint_model = load_je_model()
    app.run(debug=True, use_reloader=False)
