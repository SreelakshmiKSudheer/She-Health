import pytest
from pydantic import ValidationError

from app.schemas.questionnaire import QuestionCreate


def test_yes_no_requires_direct_mappings_and_no_options():
	with pytest.raises(ValidationError):
		QuestionCreate(
			id="Q1",
			text="Has chronic fatigue?",
			category="General",
			q_type="yes_no",
		)


def test_single_select_requires_options():
	with pytest.raises(ValidationError):
		QuestionCreate(
			id="Q2",
			text="Cycle regularity",
			category="General",
			q_type="single_select",
			direct_mappings=[{"feature_name": "irregular_cycle", "feature_value": 1.0}],
		)


def test_multi_select_valid_shape_passes():
	q = QuestionCreate(
		id="Q3",
		text="Select symptoms",
		category="PCOS",
		q_type="multi_select",
		options=[
			{
				"text": "Acne",
				"mappings": [{"feature_name": "acne", "feature_value": 1.0}],
			},
			{
				"text": "Hair thinning",
				"mappings": [{"feature_name": "hair_thinning", "feature_value": 1.0}],
			},
		],
	)

	assert q.q_type == "multi_select"
	assert len(q.options) == 2
