from flask import Blueprint, render_template, request, redirect
import json
import os

bp = Blueprint('admin', __name__)

DATA_PATH = "app/data/questions.json"

def load_questions():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_questions(questions):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

def get_next_id(questions):
    return max([q.get("id", 0) for q in questions], default=0) + 1

@bp.route('/admin', methods=['GET'])
def admin_panel():
    all_questions = load_questions()

    # 搜索 & 标签筛选
    keyword = request.args.get("search", "").lower()
    tag_filter = request.args.get("tag", "").strip()

    questions = all_questions
    if keyword:
        questions = [q for q in questions if keyword in q['question'].lower()]
    if tag_filter:
        questions = [q for q in questions if tag_filter in q.get('tags', [])]

    # 分页
    page = int(request.args.get("page", 1))
    PER_PAGE = 10
    total = len(questions)
    pages = (total + PER_PAGE - 1) // PER_PAGE
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    current_page_questions = questions[start:end]

    # 所有标签
    all_tags = sorted(set(tag for q in all_questions for tag in q.get('tags', [])))

    return render_template("quiz_admin.html",
                           questions=current_page_questions,
                           page=page,
                           pages=pages,
                           keyword=keyword,
                           tag_filter=tag_filter,
                           all_tags=all_tags)

@bp.route('/admin/add', methods=['POST'])
def add_question():
    questions = load_questions()
    question_text = request.form['question'].strip()
    answer_text = request.form['answer'].strip()
    tags_text = request.form.get('tags', '').strip()
    tags = [t.strip() for t in tags_text.split(',') if t.strip()]

    if question_text and answer_text:
        questions.append({
            "id": get_next_id(questions),
            "question": question_text,
            "answer": answer_text,
            "tags": tags
        })
        save_questions(questions)
    return redirect('/admin')

@bp.route('/admin/delete', methods=['POST'])
def delete_question():
    question_id = int(request.form['id'])
    questions = load_questions()
    questions = [q for q in questions if q.get("id") != question_id]
    save_questions(questions)
    return redirect('/admin')

@bp.route('/admin/update', methods=['POST'])
def update_question():
    question_id = int(request.form['id'])
    new_question = request.form['question'].strip()
    new_answer = request.form['answer'].strip()
    tags_text = request.form.get('tags', '').strip()
    new_tags = [t.strip() for t in tags_text.split(',') if t.strip()]

    questions = load_questions()
    for q in questions:
        if q.get("id") == question_id:
            q['question'] = new_question
            q['answer'] = new_answer
            q['tags'] = new_tags
            break
    save_questions(questions)
    return redirect('/admin')
