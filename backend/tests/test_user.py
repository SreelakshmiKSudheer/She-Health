from app.models.user import UserProfile


def test_user_profile_bmi_is_auto_calculated():
	profile = UserProfile(
		user_id="u-1",
		age=25,
		height=160.0,
		weight=64.0,
		marital_status="single",
		family_history=False,
	)
	assert profile.bmi == 25.0


def test_user_profile_bmi_overrides_incoming_value():
	profile = UserProfile(
		user_id="u-2",
		age=30,
		height=170.0,
		weight=68.0,
		bmi=999.0,
	)
	assert profile.bmi == 23.53
