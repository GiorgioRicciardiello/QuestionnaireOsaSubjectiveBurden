"""Manuscript-artifact scripts.

Each module here regenerates one specific manuscript output (a figure,
table, or shared dataset export) from the saved model results. These are
distinct from the core pipeline entry points (``train_xgboost``,
``train_tabpfn``, ``analyze_xgboost``, ``analyze_tabpfn``), which remain at
the top level of ``questionnaire_osa``.

Entry points (all use ``__main__``)
------------------------------------
* ``python -m questionnaire_osa.manuscript.make_table_one``
* ``python -m questionnaire_osa.manuscript.make_figure1``
* ``python -m questionnaire_osa.manuscript.make_figure2``
* ``python -m questionnaire_osa.manuscript.make_decision_curve``
* ``python -m questionnaire_osa.manuscript.make_reference_tables``
* ``python -m questionnaire_osa.manuscript.make_shared_dataset``
"""
