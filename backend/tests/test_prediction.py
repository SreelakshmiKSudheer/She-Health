from app.ml.predictor import SheHealthPredictor


class _ModelWithRiskCategory:
	def __init__(self):
		self.last_df = None

	def predict_proba(self, df):
		self.last_df = df.copy()
		return [[0.15, 0.85]]

	def predict_risk_category(self, df):
		return [4]


class _ModelWithOnlyProba:
	def __init__(self):
		self.last_df = None

	def predict_proba(self, df):
		self.last_df = df.copy()
		return [[0.6, 0.4]]


class _ModelWithOnlyPredict:
	def __init__(self):
		self.last_df = None

	def predict(self, df):
		self.last_df = df.copy()
		return [1]


def test_predictor_includes_all_required_diseases(monkeypatch):
	def _fake_loader(self, filenames):
		first = filenames[0]
		if first == "pcos_model.pkl":
			return {
				"model": _ModelWithRiskCategory(),
				"features": ["f"],
				"threshold": 0.25,
			}
		if first == "endometriosis_model.pkl":
			return {
				"model": _ModelWithOnlyProba(),
				"features": ["f"],
				"threshold": 0.35,
			}
		if first == "cervical_cancer_model.pkl":
			return {
				"model": _ModelWithOnlyPredict(),
				"features": ["f"],
				"threshold": 0.10,
			}
		return None

	monkeypatch.setattr(SheHealthPredictor, "_load_first_available_model", _fake_loader)
	predictor = SheHealthPredictor()
	out = predictor.predict_all_diseases({"f": 1})

	assert set(out.keys()) == {"PCOS", "Endometriosis", "Cervical"}
	assert out["PCOS"]["threshold"] == 0.25


def test_predictor_fallback_probability_to_level(monkeypatch):
	def _fake_loader(self, filenames):
		first = filenames[0]
		if first == "pcos_model.pkl":
			return {
				"model": _ModelWithOnlyProba(),  # 40% => risk level 2
				"features": ["a", "b"],
				"threshold": 0.50,
			}
		return None

	monkeypatch.setattr(SheHealthPredictor, "_load_first_available_model", _fake_loader)
	predictor = SheHealthPredictor()
	out = predictor.predict_all_diseases({"a": 1})

	assert out["PCOS"]["probability"] == 40.0
	assert out["PCOS"]["risk_level"] == 2
	assert out["PCOS"]["label"] == "Low"
	assert out["PCOS"]["threshold"] == 0.25
	assert out["PCOS"]["predicted_positive"] is True
	assert out["PCOS"]["feature_coverage"] == {
		"provided": 1,
		"expected": 2,
		"missing": 1,
		"missing_features": ["b"],
	}


def test_predictor_uses_exact_feature_order_for_model(monkeypatch):
	model = _ModelWithRiskCategory()

	def _fake_loader(self, filenames):
		if filenames[0] == "pcos_model.pkl":
			return {
				"model": model,
				"features": ["A Feature", "Another Feature"],
				"threshold": 0.25,
			}
		return None

	monkeypatch.setattr(SheHealthPredictor, "_load_first_available_model", _fake_loader)
	predictor = SheHealthPredictor()
	predictor.predict_all_diseases({"a   feature": 1.0, "another feature": 0.0, "extra": 5})

	assert list(model.last_df.columns) == ["A Feature", "Another Feature"]
	assert float(model.last_df.iloc[0]["A Feature"]) == 1.0


def test_predictor_feature_coverage_report(monkeypatch):
	def _fake_loader(self, filenames):
		if filenames[0] == "pcos_model.pkl":
			return {
				"model": _ModelWithOnlyProba(),
				"features": ["f1", "f2"],
				"threshold": 0.25,
			}
		return None

	monkeypatch.setattr(SheHealthPredictor, "_load_first_available_model", _fake_loader)
	predictor = SheHealthPredictor()
	report = predictor.get_feature_coverage_report({"f1": 1})

	assert report["PCOS"]["status"] == "ok"
	assert report["PCOS"]["feature_coverage"]["missing_features"] == ["f2"]
