from sqlalchemy.orm import Session
from app.models.questionnaire import UserResponse
from app.schemas.response import SubmitResponse

class ResponseService:
    def __init__(self, db: Session):
        self.db = db

    async def save_user_responses(self, data: SubmitResponse):
        """
        Saves the user's answers. If they have answered these 
        questions before, we replace them to keep the profile current.
        """
        user_id = data.user_id
        
        for item in data.responses:
            # 1. Clear old responses for this specific question to avoid duplicates
            self.db.query(UserResponse).filter(
                UserResponse.user_id == user_id,
                UserResponse.question_id == item.question_id
            ).delete()

            # 2. Save each selected option (handles multi-select)
            for opt_id in item.selected_option_ids:
                new_resp = UserResponse(
                    user_id=user_id,
                    question_id=item.question_id,
                    option_id=opt_id
                )
                self.db.add(new_resp)
        
        self.db.commit()
        return {"status": "success", "message": "Responses saved successfully"}

    def get_user_answers_as_feature_dict(self, user_id: str):
        """
        This is the "Bridge" function for the ML model.
        It converts the user's saved choices into a dictionary of 125 features.
        """
        from app.models.questionnaire import OptionFeatureMap
        
        # Initialize all features to 0.0 (or a neutral baseline)
        # In a real scenario, you'd pull the full list of 125 feature names
        feature_vector = {} 

        # Join UserResponse with OptionFeatureMap to get the 'Intelligence'
        results = self.db.query(OptionFeatureMap.feature_name, OptionFeatureMap.feature_value).\
            join(UserResponse, UserResponse.option_id == OptionFeatureMap.option_id).\
            filter(UserResponse.user_id == user_id).all()

        for f_name, f_val in results:
            feature_vector[f_name] = f_val
            
        return feature_vector