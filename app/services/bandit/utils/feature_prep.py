import numpy as np
from typing import List
from wordfreq import zipf_frequency

from app.schemas.bandit import BanditArmRead


class FeatureTransformer:
    def __init__(self, arms: List[BanditArmRead]):
        self.arms = arms
        self.models = sorted(list(set( arm.model_name for arm in self.arms )))
        self.prompts = sorted(list(set( arm.system_prompt for arm in self.arms )))
        # useful mapper for building feature vector
        self.model_idx = {m: i for i, m in enumerate(self.models)}
        self.prompt_idx = {p: i for i, p in enumerate(self.prompts)}
        # hour sin/cos, weekend, complexity interaction per model
        self.dimensions = len(self.models) + len(self.prompts) + (len(self.models) * 3)

    def get_feature_names(self):
        names = []
        names += [f"model_{m}" for m in self.models]
        names += [f"prompt_{p}" for p in self.prompts]
        names += [f"hour_sin_x_model_{m}" for m in self.models]
        names += [f"hour_cos_x_model_{m}" for m in self.models]
        names += [f"is_weekend_x_model_{m}" for m in self.models]
        # names += [f"complexity_x_model_{m}" for m in self.models]
        return names

    def transform_to_vector(self, arm: BanditArmRead, context: dict):
        model_vec = [0.0] * len(self.models)
        if arm.model_name in self.model_idx:
            model_vec[self.model_idx[arm.model_name]] = 1.0
            
        prompt_vec = [0.0] * len(self.prompts)
        if arm.system_prompt in self.prompt_idx:
            prompt_vec[self.prompt_idx[arm.system_prompt]] = 1.0
        # TODO: update context to user input -> score complexity here??
        hour = context.get('hour_of_day', 0)
        hour_sin_x_model = np.sin(2 * np.pi * hour / 24) * np.array(model_vec)
        hour_cos_x_model = np.cos(2 * np.pi * hour / 24) * np.array(model_vec)
        is_weekend_x_model = context.get('is_weekend', 0) * np.array(model_vec)
        # complexity_x_model = context.get('input_complexity', 0.5) * np.array(model_vec)
        
        return np.array(
            model_vec + prompt_vec + hour_sin_x_model.tolist() + hour_cos_x_model.tolist() +
            is_weekend_x_model.tolist()
            #  + complexity_x_model.tolist()
        )

    def get_report(self, theta_hat, A):
        """
        Generates a human-readable summary of weights and data volume.
        Useful for feeding your Drift Monitor and DB logs.
        """
        names = self.get_feature_names()
        if len(theta_hat) != len(names):
            return {"error": "Dimension mismatch between weights and mapper."}

        A_diag = np.diag(A)
        report = {}
        for i, name in enumerate(names):
            report[name] = {
                "weight": round(float(theta_hat[i]), 4),
                "certainty": round(float(A_diag[i]), 2)
            }
        return report

    import numpy as np

    def get_report_v2(self, theta_hat, A):
        names = self.get_feature_names()
        # A_inv diagonal gives us the variance (uncertainty) of each feature
        # Lower variance = Higher confidence
        try:
            A_inv_diag = np.diag(np.linalg.inv(A))
        except:
            A_inv_diag = np.ones(len(names)) * 999 # Handle singular matrix

        report = {"models": {}, "environment": {}}
        
        # We group by model to see the "Total Impact"
        for m in self.models:
            # Find indices for all features related to this specific model
            m_indices = [i for i, name in enumerate(names) if f"model_{m}" in name]
            
            m_report = {}
            for idx in m_indices:
                name = names[idx]
                # Strip the model name for a cleaner nested report
                short_name = name.replace(f"_x_model_{m}", "").replace(f"model_{m}", "base_intercept")
                
                m_report[short_name] = {
                    "weight": round(float(theta_hat[idx]), 4),
                    "uncertainty": round(float(A_inv_diag[idx]), 6),
                    "data_volume": round(float(np.diag(A)[idx]), 2) # Raw precision
                }
            
            report["models"][m] = m_report

        # Global features (like prompts, if shared, or global context)
        global_indices = [i for i, name in enumerate(names) if not any(m in name for m in self.models)]
        for idx in global_indices:
            report["environment"][names[idx]] = {
                "weight": round(float(theta_hat[idx]), 4),
                "uncertainty": round(float(A_inv_diag[idx]), 6) # 
            }

        return report



# def get_bandit_complexity_feature(text: str):
#     tokens = text.lower().split()
#     token_len = len(tokens)
#     if token_len == 0: return 0.0

#     # 1. VOLUME COMPONENT (Log-Saturated)
#     # Reaches 0.5 at 100 tokens, approaches 1.0 slowly after.

#     volume_score = token_len / (token_len + 100)

#     # 2. SEMANTIC DIVERSITY (Zipf Variance as placeholder)
#     # We look at the spread of word rarities.
#     # Single topic = consistent rarity. Multi-topic = erratic rarity.
#     freqs = [zipf_frequency(w, 'en') for w in tokens if len(w) > 3] # Filter short filler words
    
#     if len(freqs) > 1:
#         topic_variance = min(np.std(freqs), 2.0) / 2.0
#     else:
#         topic_variance = 0

#     return (volume_score * 0.9) + (topic_variance * 0.1)


# if __name__ == '__main__':
#     single_topic_long = "Explain Thompson Sampling"
#     multi_topic_short = """
#     I am working on four things, too many! My main job, this multi-armed bandit project, consulting and a then a pricing tool for grocers.

#     Beyond that, I think baking sourdough bread would be a very good idea. How much more do I need to type to move query complexity to a 0.5+
#     """

#     print(f"Long/Single: {get_bandit_complexity_feature(single_topic_long)}") # ~0.13
#     print(f"Short/Multi: {get_bandit_complexity_feature(multi_topic_short)}") 
