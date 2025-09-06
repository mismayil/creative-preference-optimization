#!/bin/bash

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_t0.7_p0.95' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10/en_t0.7_p0.95' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div/en_t0.7_p0.95' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov/en_t0.7_p0.95' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur/en_t0.7_p0.95' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre/en_t0.7_p0.95' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_sr_t0.7_p0.95' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_sr_t0.7_p0.95' \
#         'experiments/lm/results/gemini-2.0-flash/en_sr_t0.7_p0.95' \
#         'experiments/lm/results/gpt-4o/en_sr_t0.7_p0.95' \
#         'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_t0.8_p0.97' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10/en_t0.8_p0.97' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div/en_t0.8_p0.97' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov/en_t0.8_p0.97' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur/en_t0.8_p0.97' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre/en_t0.8_p0.97' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_sr_t0.8_p0.97' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_sr_t0.8_p0.97' \
#         'experiments/lm/results/gemini-2.0-flash/en_sr_t0.8_p0.97' \
#         'experiments/lm/results/gpt-4o/en_sr_t0.8_p0.97' \
#         'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_t0.9_p0.99' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10/en_t0.9_p0.99' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div/en_t0.9_p0.99' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov/en_t0.9_p0.99' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur/en_t0.9_p0.99' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre/en_t0.9_p0.99' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_sr_t0.9_p0.99' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_sr_t0.9_p0.99' \
#         'experiments/lm/results/gemini-2.0-flash/en_sr_t0.9_p0.99' \
#         'experiments/lm/results/gpt-4o/en_sr_t0.9_p0.99' \
#         'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_t0.7_k50' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10/en_t0.7_k50' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div/en_t0.7_k50' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov/en_t0.7_k50' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur/en_t0.7_k50' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre/en_t0.7_k50' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_sr_t0.7_k50' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_sr_t0.7_k50' \
#         'experiments/lm/results/gemini-2.0-flash/en_sr_t0.7_k50' \
#         'experiments/lm/results/gpt-4o/en_sr_t0.7_k50' \
#         -o experiments/data/cap_eval/heldout_item_human_eval \
#         -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#         -np 1

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30_5565b28b76f3.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10_90e51727b6e4.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div_a2b638eb3d83.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov_9471afcbbc3d.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur_f3fad61c19e9.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre_a89174aa5587.json' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_sr/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_short_response_llama-3.1-8b-instruct_81bad05178a5.json' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_sr/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_short_response_claude-3-7-sonnet-20250219_c36adec41519.json' \
#         'experiments/lm/results/gemini-2.0-flash/en_sr/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_short_response_gemini-2.0-flash_bda5264777af.json' \
#         'experiments/lm/results/gpt-4o/en_sr/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_short_response_gpt-4o_42984bc21a48.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov_00df49584e01.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11_df3d481784ec.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov_1318f7425e86.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30_19d225876de2.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov_45e56feea7b6.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er_a139f9fadac7.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs_bf8c0cff6be7.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua/en/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default_cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua_549313f36c42.json' \
#         -o experiments/data/cap_eval/heldout_item_all \
#         -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#         -np 1

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre/en_heldout16' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_scr_heldout16' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_scr_heldout16' \
#         'experiments/lm/results/gemini-2.0-flash/en_scr_heldout16' \
#         'experiments/lm/results/gpt-4o/en_scr_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre-qua/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-fa-msd5-mm10-nov/en_heldout16' \
#         -o experiments/data/cap_eval/heldout_item16 \
#         -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#         -np 1


# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-div/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-sur/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-cre/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-qua/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-cre-qua/en_heldout16' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_scr_heldout16' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_scr_heldout16' \
#         'experiments/lm/results/gemini-2.0-flash/en_scr_heldout16' \
#         'experiments/lm/results/gpt-4o/en_scr_heldout16' \
#         -o experiments/data/cap_eval/heldout_item16_ms11 \
#         -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#         -np 1

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-div/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-sur/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-cre/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-cre-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/gemini-2.0-flash/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/gpt-4o/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         -o experiments/data/cap_eval/test_stories16_ms11

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_heldout16' \
#         'experiments/lm/results/cpo-sft-llama-3.1-8b-tulu3-lora-fa-ms30/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-tulu3-lora-fa-msd5-mm10-ms11-nov/en_heldout16' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_scr_heldout16' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_scr_heldout16' \
#         'experiments/lm/results/gemini-2.0-flash/en_scr_heldout16' \
#         'experiments/lm/results/gpt-4o/en_scr_heldout16' \
#         -o experiments/data/cap_eval/heldout_item16_tulu3 \
#         -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#         -np 1

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-ms30/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-nov/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-mult-cre/en_heldout16' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-mistral-small-24b-instruct-lora-fa-msd5-mm10-ms11/en_heldout16' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_scr_heldout16' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_scr_heldout16' \
#         'experiments/lm/results/gemini-2.0-flash/en_scr_heldout16' \
#         'experiments/lm/results/gpt-4o/en_scr_heldout16' \
#         -o experiments/data/cap_eval/heldout_item16_new \
#         -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#         -np 1

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-ms30/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-nov/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-mult-cre/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/cpo-dpo-sft-ms30-mistral-small-24b-instruct-lora-fa-msd5-mm10-ms11/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/llama-3.1-8b-instruct/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/gemini-2.0-flash/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         'experiments/lm/results/gpt-4o/en_test_stories/cpo_en_stories_text_test_eval_data_default.json' \
#         -o experiments/data/cap_eval/test_stories16_new

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/human-ref/cpo_en_stories_text_train_eval_data_default.json' \
#         -o experiments/data/cap_eval/test_stories16_new \
#         -sf _human_ref

# python prepare_ext_eval_data.py \
#         -i "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-nov-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-mult-nov-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-mult-nov-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-mult-nov-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-mult-nov-qua/en_test_stories/cpo_en_stories_text_test_eval_data_default.json" \
#         -o experiments/data/cap_eval/test_stories16_new \
#         -sf _new_cpo

# python prepare_ext_eval_data.py \
#         -i "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov-qua/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov-qua/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-nov-qua/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-mult-nov-qua/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-mult-nov-qua/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-mult-nov-qua/en_heldout16" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-mult-nov-qua/en_heldout16" \
#         -o experiments/data/cap_eval/heldout_item16_new \
#         -sf _new_cpo

baselines=(
    "claude-3-7-sonnet-20250219"
    "gpt-4o"
    "gemini-2.0-flash"
    "meta-llama/Llama-3.1-8B-Instruct"
)

final_cpo_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
)

models=(
    ${baselines[@]}
    ${final_cpo_models[@]}
)

# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     python prepare_ext_eval_data.py \
#         -i experiments/lm/results/${model_name}/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json \
#         -o experiments/data/ocsai_eval/test_sent_comp16 \
#         -m ocsai \
#         -sf _${model_name}
# done

# python prepare_ext_eval_data.py \
#         -i 'experiments/lm/results/llama-3.1-8b-instruct/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json' \
#         'experiments/lm/results/claude-3-7-sonnet-20250219/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json' \
#         'experiments/lm/results/gemini-2.0-flash/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json' \
#         'experiments/lm/results/gpt-4o/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json' \
#         "experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov-qua/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-div/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-sur/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre-qua/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua/en_sent_comp/cpo_en_sent_comp_text_test_eval_data_default.json" \
#         -o experiments/data/ocsai_eval/test_sent_comp16 \
#         -m ocsai

python prepare_ext_eval_data.py \
        -i 'experiments/lm/results/human-ref/cpo_en_sent_comp_text_train_eval_data_default.json' \
        -o experiments/data/ocsai_eval/test_sent_comp16 \
        -sf _human_ref \
        -m ocsai
