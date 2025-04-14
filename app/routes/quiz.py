from flask import Blueprint, render_template, request, session, redirect, url_for
import json
import random

bp = Blueprint('quiz', __name__)
DATA_PATH = "app/data/questions.json"

@bp.route('/quiz', methods=['GET'])
def quiz():
    # 加载题库
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # 初始化 session 中的已出题 id 列表以及答题统计
    if 'used_ids' not in session:
        session['used_ids'] = []
    if 'correct_count' not in session:
        session['correct_count'] = 0
    if 'incorrect_count' not in session:
        session['incorrect_count'] = 0
    if 'total_count' not in session:
        session['total_count'] = 0

    used_ids = session['used_ids']
    
    available_questions = [q for q in questions if q["id"] not in used_ids]

    # 如果全部题目都已出过，则重置
    if not available_questions:
        session['used_ids'] = []
        available_questions = questions

    # 随机抽取未出过的题
    question = random.choice(available_questions)
    session['current_question'] = question  # 保留当前题目
    session.modified = True
    result = None
    if request.method == 'POST':
        user_answer = request.form['answer'].strip()
        correct = user_answer.lower() == question['answer'].strip().lower()

        # 更新答题统计
        if correct:
            session['correct_count'] += 1
        else:
            session['incorrect_count'] += 1

        session['total_count'] += 1
        session['used_ids'].append(question["id"])
        session.modified = True  # 通知 Flask 更新 session

        result = "✅ 正确！" if correct else f"❌ 错误，正确答案是：{question['answer']}"
        
        # 跳转到下一题
        return redirect(url_for('quiz.quiz'))

    return render_template("quiz.html", question=question, result=result, stats={
        'total_count': session['total_count'],
        'correct_count': session['correct_count'],
        'incorrect_count': session['incorrect_count']
    })

@bp.route('/quiz', methods=['POST'])
def submit_answer():
    question = session.get("current_question")
    if not question:
        return redirect(url_for("quiz.get_question"))
    if 'answer' in request.form:
        user_answer = request.form['answer'].strip()
        correct = user_answer.lower() == question['answer'].strip().lower()
    else:
        correct = False
        

    # 统计答题
    if correct:
        session['correct_count'] += 1
    else:
        session['incorrect_count'] += 1
    session['total_count'] += 1
    session['used_ids'].append(question["id"])
    session.modified = True

    result = "✅ 正确！" if correct else f"❌ 错误，正确答案是：{question['answer']}"

    return render_template("quiz.html", question=question, result=result, stats={
        "correct_count": session['correct_count'],
        "total_count": session['total_count'],
        'incorrect_count': session['incorrect_count']
    })