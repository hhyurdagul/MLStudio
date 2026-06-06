import streamlit as st
import polars as pl

def render_preprocessing_component(
    preprocessing_df: pl.DataFrame,
) -> pl.DataFrame:
    is_customized = st.checkbox("Customize preprocessing")

    st.write("Selected Preprocessing Steps")
    if is_customized:
        with st.container(border=True):
            st.write("Customize Encoding")
            st.caption("Click on preprocessing values to make changes")
            _temp_df_1 = st.data_editor(
                preprocessing_df.filter(pl.col("Type") == "String"),
                disabled=["Variable", "Type", "Unique Count"],
                column_order=["Variable", "Preprocessing"],
                column_config={
                    "Preprocessing": st.column_config.SelectboxColumn(
                        help="Preprocessing type of variable",
                        width="medium",
                        options=["OrdinalEncoder", "OneHotEncoder"],
                        required=True,
                    )
                },
            )

            st.write("Customize Scaling")
            st.caption("Click on preprocessing values to make changes")
            _temp_df_2 = st.data_editor(
                preprocessing_df.filter(pl.col("Type") == "Numeric"),
                disabled=["Variable", "Type", "Unique Count"],
                column_order=["Variable", "Preprocessing"],
                column_config={
                    "Preprocessing": st.column_config.SelectboxColumn(
                        help="Preprocessing type of variable",
                        width="medium",
                        options=["StandardScaler", "MinMaxScaler", "None"],
                        required=True,
                    )
                },
            )

            preprocessing_df = _temp_df_1.vstack(_temp_df_2)

    else:
        st.dataframe(preprocessing_df)
    return preprocessing_df



