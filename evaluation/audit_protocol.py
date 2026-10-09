"""Rescore and compare completed tool-adapter predictions without a GPU."""
import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.benchmark import parse_calls,score_calls,summarize,file_sha256
from evaluation.checkpoint import write_json_atomic
from runtime.mock_tools import MockWorld


def norm(value):
    return ''.join(c.casefold() for c in value if c.isalnum()) if isinstance(value,str) else value


def audit(adapter,previous):
    folder = adapter/'evaluation'
    manifest = json.loads((adapter/'training_manifest.json').read_text(encoding='utf-8'))
    fresh = manifest.get('fresh_base',False)
    config = manifest['config']
    model_label = f"Fresh r{config['lora_r']} ({manifest['train_rows']:,} training rows)" if fresh else 'New continuation'
    test_path = adapter/'held_out_test.jsonl'
    rows = [json.loads(line) for line in test_path.read_text(encoding='utf-8').splitlines()]
    predictions = folder/'test-961-tokens256-predictions.jsonl'
    records = [json.loads(line) for line in predictions.read_text(encoding='utf-8').splitlines()]
    indices = [r['test_index'] for r in records]
    if len(records) != len(rows) or len(set(indices)) != len(rows) or set(indices) != set(range(len(rows))):
        raise ValueError('Prediction coverage is incomplete or duplicated.')
    old_path = previous/'evaluation/test-961-tokens256-predictions.jsonl'
    if not old_path.exists():
        old_path = previous/'evaluation/benchmark-961-seed42-tokens256-predictions.jsonl'
    old_records = [r for r in (json.loads(line) for line in old_path.read_text(encoding='utf-8').splitlines()) if r['variant']=='lora']
    old = {r['test_index']:r for r in old_records}
    if ((previous/'held_out_test.jsonl').read_bytes() != test_path.read_bytes() or len(old_records) != len(rows)
        or set(old) != set(range(len(rows)))):
        raise ValueError('Previous model was evaluated on different test cases.')
    old_manifest_path = previous/'training_manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text(encoding='utf-8')) if old_manifest_path.exists() else {}
    common_keys = ('model_id','lora_r','lora_alpha','lora_dropout','target_modules','head_lr','transformer_lr',
                   'gradient_accumulation_steps','epochs','max_length','seed')
    same_fresh_recipe = (fresh and old_manifest.get('fresh_base',False)
                         and manifest['label_policy'] == old_manifest['label_policy']
                         and manifest['base_files'] == old_manifest['base_files']
                         and all(config.get(k) == old_manifest['config'].get(k) for k in common_keys))
    previous_label = (f"Fresh r{old_manifest['config']['lora_r']} ({old_manifest['train_rows']:,} training rows)"
                      if old_manifest.get('fresh_base') else 'Earlier full-data repair')
    errors,categories,fields = [],Counter(),Counter()
    groups = {'single_call':[],'multiple_calls':[]}
    transitions = Counter()
    for record in records:
        row = rows[record['test_index']]
        expected = [{'name':c['function']['name'],'arguments':c['function']['arguments']} for c in row['messages'][-1]['tool_calls']]
        if record['expected_calls'] != expected or record['prompt'] != row['messages'][-2]['content']:
            raise ValueError('Saved question/reference changed.')
        predicted,error = parse_calls(record['output_text'])
        scores = score_calls(predicted,expected,error)
        if predicted != record['predicted_calls'] or error != record['parse_error']:
            raise ValueError('Saved parsed calls differ from current parsing.')
        if any(record[key] != value for key,value in scores.items()):
            raise ValueError('Saved scores differ from current scoring.')
        world = MockWorld(row['tools'])
        execution = world.execute(predicted) if error is None else {'ok':False,'errors':[{'code':'parse','message':error}]}
        if execution['ok'] != record['mock_execution_ok']:
            raise ValueError('Saved mock score differs from current execution.')
        prior = old[record['test_index']]
        if prior['expected_calls'] != expected or prior['prompt'] != record['prompt']:
            raise ValueError('Previous question/reference changed.')
        old_calls,old_error = parse_calls(prior['output_text'])
        if any(prior[key] != value for key,value in score_calls(old_calls,expected,old_error).items()):
            raise ValueError('Previous saved scores differ from current scoring.')
        groups['multiple_calls' if len(expected)>1 else 'single_call'].append(record)
        transition = ('both_exact' if prior['full_call_exact'] and record['full_call_exact'] else
                      'new_only_exact' if record['full_call_exact'] else
                      'old_only_exact' if prior['full_call_exact'] else 'neither_exact')
        transitions[transition]+=1
        if record['full_call_exact'] and execution['ok']:
            continue
        differences = []
        if error:
            category = 'invalid_format'
        elif not record['tools_correct']:
            category = 'tool_selection_or_count'
        else:
            aligned=[]
            for name in sorted(set(c['name'] for c in expected)):
                gold = [c for c in expected if c['name']==name]
                guesses = [c for c in predicted if c['name']==name]
                selected = min(permutations(guesses),key=lambda sequence:sum(sum(a['arguments'].get(k)!=b['arguments'].get(k) for k in a['arguments'].keys()|b['arguments'].keys()) for a,b in zip(gold,sequence)))
                aligned.extend(zip(gold,selected))
            for a,b in aligned:
                ea,pa = a['arguments'],b['arguments']
                for key in sorted(ea.keys()|pa.keys()):
                    if key not in ea or key not in pa or ea[key] != pa[key]:
                        fields[f"{a['name']}.{key}"]+=1
                        differences.append({'tool':a['name'],'field':key,'expected':ea.get(key),'predicted':pa.get(key),'missing':key not in pa,'extra':key not in ea})
            category = ('exact_but_mock_rejected' if not differences else
                        'only_case_spacing_punctuation' if all(isinstance(d['expected'],str) and isinstance(d['predicted'],str) and norm(d['expected'])==norm(d['predicted']) for d in differences) else
                        'datetime_difference' if any(d['field']=='datetime' for d in differences) else 'other_argument_difference')
        categories[category]+=1
        errors.append({'test_index':record['test_index'],'category':category,
                       'expected_call_count':len(expected),'prompt':record['prompt'],
                       'system_context':row['messages'][0]['content'],'expected_calls':expected,
                       'predicted_calls':predicted,'output_text':record['output_text'],'parse_error':error,
                       'differences':differences,'mock_execution_ok':execution['ok'],'mock_errors':execution['errors'],
                       'old_full_call_exact':prior['full_call_exact'],'new_full_call_exact':record['full_call_exact']})
    result = {'coverage_verified':True,'examples':len(rows),'adapter':str(adapter),'fresh_base':fresh,
              'previous_adapter':str(previous),'same_fresh_recipe':same_fresh_recipe,
              'test_sha256':file_sha256(test_path),
              'prediction_sha256':file_sha256(predictions),'old_prediction_sha256':file_sha256(old_path),
              'new_metrics':summarize(records),'old_metrics':summarize(old_records),
              'new_counts':{key:sum(r[key] for r in records) for key in ('valid_tool_call','tools_correct','full_call_exact','mock_execution_ok','hit_token_limit')},
              'paired_exact_transitions':dict(transitions),'error_categories':dict(categories),
              'differing_argument_fields':dict(fields),'by_expected_call_count':{},'errors':errors}
    for label,group in groups.items():
        result['by_expected_call_count'][label] = {**summarize(group),'mock_execution_success_rate':sum(r['mock_execution_ok'] for r in group)/len(group)}
    write_json_atomic(folder/'test-error-audit.json',result)
    lines = ['# Completed tool adapter: independent score audit','',
             f"Verified {len(rows)} unique records against their frozen questions, references and fresh mock execution.",'',
             f'| Metric | {previous_label} | {model_label} |','|---|---:|---:|']
    for key,label in (('valid_tool_call_rate','Valid format'),('tool_selection_accuracy','Tools/counts'),('full_call_exact_match_rate','Exact full calls')):
        lines.append(f"| {label} | {result['old_metrics'][key]:.2%} | {result['new_metrics'][key]:.2%} |")
    net = transitions['new_only_exact']-transitions['old_only_exact']
    lines += ['',f"Exact-match changes: {transitions['new_only_exact']} newly correct; {transitions['old_only_exact']} previously correct cases lost; net {net:+d} cases.",
              ('Both adapters use the same fresh-Base recipe. The data-size study fixes epochs, not compute; this paired audit describes the saved test outcomes.'
               if same_fresh_recipe else 'These models had different training histories; this is not a controlled rank or data-size comparison.'),
              'A score difference on this one evaluation does not establish general superiority; compare format, mock reliability and repeated seeds.',
              ('The fresh adapter was selected by internal validation. This audit does not use test scores to change that selection.' if fresh
               else 'The continuation is a repair experiment; keep its training history separate from fresh-rank conditions.'),'',
              '| Error category | Cases |','|---|---:|']
    for category,count in categories.items():
        lines.append(f'| {category} | {count} |')
    lines += ['', '## Single versus multiple expected calls','', '| Group | Cases | Valid format | Exact |','|---|---:|---:|---:|']
    for label,value in result['by_expected_call_count'].items():
        lines.append(f"| {label} | {value['examples']} | {value['valid_tool_call_rate']:.2%} | {value['full_call_exact_match_rate']:.2%} |")
    lines += ['', 'Case/spacing/punctuation is a literal review category, not automatic semantic correctness.',
              'Raw outputs, reference arguments and mock rejection details are in test-error-audit.json.',
              'Official test examples must not be copied into the next training dataset.','']
    (folder/'test-error-audit.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--adapter',type=Path,default=ROOT/'outputs/protocol-repair/r16-joint-completion')
    cli.add_argument('--previous',type=Path,default=ROOT/'outputs/qwen3-8b-base-tool-lora-r16-head-repair-joint')
    args = cli.parse_args()
    adapter = args.adapter.resolve()
    if not adapter.is_relative_to(ROOT):
        raise ValueError('Audit output must stay inside the project.')
    result = audit(adapter,args.previous.resolve())
    print(json.dumps({key:result[key] for key in ('coverage_verified','new_counts','paired_exact_transitions','error_categories','by_expected_call_count')},indent=2))
    print('Invalid format indices:',[r['test_index'] for r in result['errors'] if r['category']=='invalid_format'])


if __name__ == '__main__':
    main()
