import numpy as np

# This class return probability of being Earthquake
class pb_SAM:
    """
    This class return probability of being an Earthquake
    Copied from Pablo Lara's real_time.functions.utils()


    """

    def predict(self, X):
        meta_features = np.column_stack(
            [
                np.column_stack([model.predict(X) for model in base_models]).mean(
                    axis=1
                )
                for base_models in self.base_models_
            ]
        )
        return self.meta_model_.predict(meta_features)
