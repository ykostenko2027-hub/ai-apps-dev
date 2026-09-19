# Script to run comparison between 3 configurations for PR4
import json
import time
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm, schema
from app.main import load_context

FUZZY_INSTRUCTION = 'Ти помічник служби підтримки. Відповідай на запитання клієнта.'

def call_with_retry(fn, *args, **kwargs):
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            err_str = str(exc)
            if '429' in err_str or 'Rate' in type(exc).__name__ or 'RESOURCE_EXHAUSTED' in err_str:
                print(f'  [429 RateLimit] Сплю 9 с перед повтором (спроба {attempt+1}/6)...')
                time.sleep(9)
            else:
                raise

def run():
    client = llm.get_client()
    context = load_context()
    with open('compare/requests.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    single_queries = data['звернення']
    dialogues = data['діалоги']

    results = {
        'single_queries': [],
        'dialogues': []
    }

    print('=== 1. Прогін одиночних звернень ===')
    for idx, q in enumerate(single_queries):
        text = q['текст']
        kind = q['вид']
        print(f'[{idx+1}/{len(single_queries)}] {kind}: {text[:45]}...')

        # 1. Fuzzy
        t0 = time.perf_counter()
        try:
            time.sleep(1.5)
            r1 = call_with_retry(
                client.chat.completions.create,
                model=llm.MODEL,
                messages=[
                    {'role': 'system', 'content': FUZZY_INSTRUCTION},
                    {'role': 'user', 'content': text}
                ],
                max_tokens=llm.MAX_TOKENS,
                temperature=llm.TEMPERATURE
            )
            e1 = time.perf_counter() - t0
            txt1 = r1.choices[0].message.content or ''
            usage1 = r1.usage
            try:
                schema.validate(txt1)
                valid1 = True
            except Exception:
                valid1 = False
        except Exception as err:
            e1 = time.perf_counter() - t0
            txt1 = str(err)
            usage1 = None
            valid1 = False

        # 2. Structured Prompt (No context, no history)
        t0 = time.perf_counter()
        try:
            time.sleep(1.5)
            r2 = call_with_retry(
                client.chat.completions.create,
                model=llm.MODEL,
                messages=[
                    {'role': 'system', 'content': llm.SYSTEM_INSTRUCTION},
                    {'role': 'user', 'content': text}
                ],
                max_tokens=llm.MAX_TOKENS,
                temperature=llm.TEMPERATURE,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {'name': 'SupportResponse', 'schema': schema.output_schema()}
                }
            )
            e2 = time.perf_counter() - t0
            txt2 = r2.choices[0].message.content or ''
            usage2 = r2.usage
            try:
                val2 = schema.validate(txt2)
                valid2 = True
            except Exception:
                val2 = None
                valid2 = False
        except Exception as err:
            e2 = time.perf_counter() - t0
            txt2 = str(err)
            val2 = None
            usage2 = None
            valid2 = False

        # 3. Context Engineering (Full context, schema, fit_budget)
        t0 = time.perf_counter()
        try:
            time.sleep(1.5)
            res3 = call_with_retry(llm.ask, text, [], context)
            e3 = res3['elapsed']
            val3 = res3['result']
            usage3 = res3['usage']
            valid3 = True
        except Exception as err:
            e3 = time.perf_counter() - t0
            val3 = {'error': str(err)}
            usage3 = {}
            valid3 = False

        results['single_queries'].append({
            'kind': kind,
            'text': text,
            'fuzzy': {'raw': txt1, 'valid_schema': valid1, 'elapsed': round(e1, 2), 'usage': getattr(usage1, 'total_tokens', 0) if usage1 else 0},
            'structured': {'result': val2, 'raw': txt2, 'valid_schema': valid2, 'elapsed': round(e2, 2), 'usage': getattr(usage2, 'total_tokens', 0) if usage2 else 0},
            'context_eng': {'result': val3, 'valid_schema': valid3, 'elapsed': round(e3, 2), 'usage': usage3.get('total_tokens', 0)}
        })

    print('=== 2. Прогін діалогів ===')
    for d_idx, d in enumerate(dialogues):
        kind = d['вид']
        check_goal = d['перевіряє']
        turns = d['репліки']
        print(f'Діалог {d_idx+1}: {kind} ({len(turns)} реплік)')

        d_res = {
            'kind': kind,
            'goal': check_goal,
            'turns': []
        }

        ce_history = []

        for t_idx, turn_text in enumerate(turns):
            print(f'  Репліка {t_idx+1}: {turn_text[:45]}...')

            # 1. Fuzzy (no history)
            t0 = time.perf_counter()
            time.sleep(1.5)
            r1 = call_with_retry(
                client.chat.completions.create,
                model=llm.MODEL,
                messages=[
                    {'role': 'system', 'content': FUZZY_INSTRUCTION},
                    {'role': 'user', 'content': turn_text}
                ],
                max_tokens=llm.MAX_TOKENS,
                temperature=llm.TEMPERATURE
            )
            e1 = time.perf_counter() - t0
            txt1 = r1.choices[0].message.content or ''
            try:
                schema.validate(txt1)
                v1 = True
            except Exception:
                v1 = False

            # 2. Structured (no context, no history)
            t0 = time.perf_counter()
            time.sleep(1.5)
            r2 = call_with_retry(
                client.chat.completions.create,
                model=llm.MODEL,
                messages=[
                    {'role': 'system', 'content': llm.SYSTEM_INSTRUCTION},
                    {'role': 'user', 'content': turn_text}
                ],
                max_tokens=llm.MAX_TOKENS,
                temperature=llm.TEMPERATURE,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {'name': 'SupportResponse', 'schema': schema.output_schema()}
                }
            )
            e2 = time.perf_counter() - t0
            txt2 = r2.choices[0].message.content or ''
            try:
                val2 = schema.validate(txt2)
                v2 = True
            except Exception:
                val2 = None
                v2 = False

            # 3. Context Engineering (with history!)
            time.sleep(1.5)
            res3 = call_with_retry(llm.ask, turn_text, ce_history, context)
            val3 = res3['result']
            e3 = res3['elapsed']
            u3 = res3['usage']
            v3 = True

            # Update history for next turns
            ce_history.append({'role': 'user', 'content': turn_text})
            ce_history.append({'role': 'assistant', 'content': val3.get('reply', '')})

            d_res['turns'].append({
                'user': turn_text,
                'fuzzy': {'raw': txt1, 'valid_schema': v1, 'elapsed': round(e1, 2), 'usage': r1.usage.total_tokens if r1.usage else 0},
                'structured': {'result': val2, 'valid_schema': v2, 'elapsed': round(e2, 2), 'usage': r2.usage.total_tokens if r2.usage else 0},
                'context_eng': {'result': val3, 'valid_schema': v3, 'elapsed': round(e3, 2), 'usage': u3.get('total_tokens', 0)}
            })

        results['dialogues'].append(d_res)

    with open('compare/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Успіх! Результати збережено в compare/results.json')

if __name__ == '__main__':
    run()
