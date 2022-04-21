import pandas as pd
import numpy as np  
#from datetime import datetime
import altair as alt

def hpt_lr_chart( metric):
    
    input_df = pd.read_csv("saved_work/hp_tuning_LogReg.csv")
    input_df['type'].replace('test','validation', inplace=True)
    input_df['type'].replace('train','training', inplace=True)

    ### C values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','C',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['C'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'C:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='C:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='C:O',
    ).transform_filter(
        nearest
    )
    chart1 = alt.layer(line, selectors, points, rules, text)

    ### penalty values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','penalty',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['penalty'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'penalty:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='penalty:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='penalty:O',
    ).transform_filter(
        nearest
    )
    chart2 = alt.layer(line, selectors, points, rules, text)


    ### solver values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','solver',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['solver'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'solver:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='solver:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='solver:O',
    ).transform_filter(
        nearest
    )
    chart3 = alt.layer(line, selectors, points, rules, text)

    output_chart = chart1 | chart2 | chart3
    output_chart = output_chart.properties(
        title={
            "text": "LR Models: {} Score For Hyperparameter Tuning".format(metric),
            "subtitle": "Median {} score (y-axis) across all iterations of the parameter (x-axis)".format(metric),
            "anchor": "start",
            "fontSize": 18,
            "subtitleFontSize": 14,
           
        })
    return output_chart

def hpt_svc_chart( metric):

    input_df = pd.read_csv("saved_work/hp_tuning_SVC_poly.csv")
    input_df = pd.concat([input_df, pd.read_csv("saved_work/hp_tuning_SVC_linear.csv")])
    input_df = pd.concat([input_df, pd.read_csv("saved_work/hp_tuning_SVC_rbf.csv")])
    input_df = pd.concat([input_df, pd.read_csv("saved_work/hp_tuning_SVC_sigmoid.csv")])
    input_df['type'].replace('test','validation', inplace=True)
    input_df['type'].replace('train','training', inplace=True)

    ### C values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','C',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['C'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'C:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='C:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='C:O',
    ).transform_filter(
        nearest
    )
    chart1 = alt.layer(line, selectors, points, rules, text)

    ### gamma values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','gamma',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['gamma'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'gamma:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='gamma:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='gamma:O',
    ).transform_filter(
        nearest
    )
    chart2 = alt.layer(line, selectors, points, rules, text)


    ### kernel values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','kernel',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['kernel'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'kernel:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='kernel:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='kernel:O',
    ).transform_filter(
        nearest
    )
    chart3 = alt.layer(line, selectors, points, rules, text)

    output_chart = chart1 | chart2 | chart3
    output_chart = output_chart.properties(
        title={
            "text": "SVC Models: {} Score For Hyperparameter Tuning".format(metric),
            "subtitle": "Median {} score (y-axis) across all iterations of the parameter (x-axis)".format(metric),
            "anchor": "start",
            "fontSize": 18,
            "subtitleFontSize": 14,
           
        })
    return output_chart

def hpt_gbdt_chart( metric):

    input_df = pd.read_csv("saved_work/hp_tuning_GBT.csv")
    input_df['type'].replace('test','validation', inplace=True)
    input_df['type'].replace('train','training', inplace=True)
    
    ### learning_rate values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','learning_rate',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['learning_rate'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'learning_rate:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='learning_rate:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='learning_rate:O',
    ).transform_filter(
        nearest
    )
    chart1 = alt.layer(line, selectors, points, rules, text)

    ### n_estimators values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','n_estimators',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['n_estimators'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'n_estimators:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='n_estimators:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='n_estimators:O',
    ).transform_filter(
        nearest
    )
    chart2 = alt.layer(line, selectors, points, rules, text)


    ### max_depth values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','max_depth',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['max_depth'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'max_depth:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='max_depth:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='max_depth:O',
    ).transform_filter(
        nearest
    )
    chart3 = alt.layer(line, selectors, points, rules, text)

    ### max_features values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','max_features',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['max_features'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'max_features:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='max_features:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='max_features:O',
    ).transform_filter(
        nearest
    )
    chart4 = alt.layer(line, selectors, points, rules, text)

    ### subsample values
    df = input_df.copy()
    df = df[df.Score==metric]
    df = df.groupby(['type','subsample',])['Result'].median().reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', clear='mouseout',
                            fields=['subsample'], empty='none')
    line = alt.Chart(df).mark_line().encode(
        x = 'subsample:O',
        y = 'Result:Q',
        color = 'type:N',
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='subsample:O',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text('Result:Q', format=",.2f"), alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='subsample:O',
    ).transform_filter(
        nearest
    )
    chart5 = alt.layer(line, selectors, points, rules, text)

    output_chart = chart1 | chart2 | chart3 | chart4 | chart5
    output_chart = output_chart.properties(
        title={
            "text": "GBDT Models: {} Score For Hyperparameter Tuning".format(metric),
            "subtitle": "Median {} score (y-axis) across all iterations of the parameter (x-axis)".format(metric),
            "anchor": "start",
            "fontSize": 18,
            "subtitleFontSize": 14,
           
        })
    return output_chart