import numpy as np
import xarray as xr


def rbf_pca_predict(df_dataset,pca,rbf):
    # Apply PCA to the dataset
    predicted_Setup_coef=rbf.predict(dataset=df_dataset)
    predicted_Setup_ds = xr.Dataset(
        {
            "PCs": (["case_num", "n_component"], predicted_Setup_coef.values),
        },
        coords={
            "case_num": np.arange(predicted_Setup_coef.values.shape[0]),
            "n_component": np.arange(predicted_Setup_coef.values.shape[1]),
        },
    )
    ds_output = pca.inverse_transform(PCs=predicted_Setup_ds)
    return ds_output
        