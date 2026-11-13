"""Frozen R3 formula uses the complete codebook, without hidden special cases."""
import pytest
from tada_taskgen.scoring import balanced_label_scores, LABEL_AGGREGATION


def cells(answers, options):
    return [{'id': f'c{i}', 'question': 'q', 'answer': answer,
             'encoding': [[a, a] for a in options]} for i, answer in enumerate(answers)]


def evaluate(answers, options, predictions, penalty=1.0):
    rows = cells(answers, options)
    contributions = {r['id']: float(expected == predicted) * penalty
                     for r, expected, predicted in zip(rows, answers, predictions)}
    return balanced_label_scores(rows, contributions)


@pytest.mark.parametrize('options', [['A', 'B'], ['A', 'B', 'C'], ['A', 'B', 'C', 'D']])
def test_constant_answers_score_zero_when_all_codebook_classes_are_present(options):
    answers = options * 3
    for answer in options:
        score, detail = evaluate(answers, options, [answer] * len(answers))
        assert score == 0
        assert detail['q']['balanced_accuracy'] == pytest.approx(1 / len(options))
        assert detail['q']['codebook_option_count'] == len(options)


def test_missing_class_does_not_shrink_the_frozen_codebook_chance_floor():
    # The brief's literal formula implies .25 here, despite its general claim
    # that every constant answer scores zero. Do not add an undocumented cap.
    answers = ['A', 'A', 'B', 'B']
    score, details = evaluate(answers, ['A', 'B', 'C'], ['A'] * 4)
    question = details['q']
    assert question['present_classes'] == ['A', 'B']
    assert question['balanced_accuracy'] == .5
    assert question['codebook_option_count'] == 3
    assert question['chance_floor'] == pytest.approx(1 / 3)
    assert score == pytest.approx(.25)


def test_below_chance_floors_at_zero_and_perfect_remains_one():
    answers = ['A', 'A', 'B', 'B']
    assert evaluate(answers, ['A', 'B'], ['B', 'B', 'A', 'A'])[0] == 0
    assert evaluate(answers, ['A', 'B'], answers)[0] == 1


def test_sheet_precision_penalty_precedes_chance_correction():
    answers = ['A', 'A', 'B', 'B']
    result, details = evaluate(answers, ['A', 'B'], answers, penalty=.75)
    assert details['q']['balanced_accuracy'] == .75
    assert result == .5  # Not .75: the penalty is inside the correction formula.


def test_question_mean_and_work_item_input_keep_original_denominators():
    q1 = cells(['A'] * 8 + ['B'] * 2, ['A', 'B'])
    q2 = [{'id': 'other-' + r['id'], 'question': 'other', **{k: v for k, v in r.items() if k not in ('id', 'question')}}
          for r in cells(['A', 'B'], ['A', 'B'])]
    contribution = {r['id']: float(r['answer'] == 'A') for r in q1}
    contribution.update({r['id']: 1.0 for r in q2})
    value, questions = balanced_label_scores(q1 + q2, contribution)
    assert value == .5
    assert questions['q']['class_counts'] == {'A': 8, 'B': 2}
    assert questions['q']['score'] == 0
    assert questions['other']['score'] == 1
    assert LABEL_AGGREGATION == 'per_question_chance_corrected_balanced_accuracy'


@pytest.mark.parametrize('mutation', ['single', 'duplicate_answer', 'duplicate_visible', 'different', 'missing_answer'])
def test_invalid_or_inconsistent_codebooks_fail_closed(mutation):
    rows = cells(['A', 'B'], ['A', 'B'])
    if mutation == 'single':
        rows[0]['encoding'] = [['A', 'A']]
    elif mutation == 'duplicate_answer':
        rows[0]['encoding'] = [['A', 'A'], ['A', 'B']]
    elif mutation == 'duplicate_visible':
        rows[0]['encoding'] = [['A', 'A'], ['B', 'A']]
    elif mutation == 'different':
        rows[1]['encoding'].append(['C', 'C'])
    else:
        for row in rows:
            row['encoding'] = [['A', 'A'], ['C', 'C']]
    with pytest.raises(ValueError):
        balanced_label_scores(rows, {r['id']: 1 for r in rows})
