import streamlit as st

from mlstudio.components import (
    render_data_component,
    render_input_training_data_component,
    render_modeling_component,
    render_preprocessing_component,
)
from mlstudio.ml.data import (
    create_transformer_and_transform_data,
    get_preprocessing_data,
    read_data,
)


df, features, target = render_input_training_data_component(read_data)
if df is not None and target is not None:
    with st.expander("Data"):
        render_data_component(df, features, target)

    if not features:
        st.info("Select at least one feature to configure preprocessing and modeling.")
    else:
        with st.expander("Preprocessing"):
            preprocessing_df = render_preprocessing_component(
                get_preprocessing_data(df.select(features))
            )
            transformer, transformed_data = create_transformer_and_transform_data(
                df.select(features),
                preprocessing_df,
            )
            st.dataframe(transformed_data)

        with st.expander("Modeling", expanded=True):
            render_modeling_component(df, features, target, transformer)
