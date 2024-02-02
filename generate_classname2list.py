import os
import sys
import math
import pickle
import random
from yacs.config import CfgNode as CN
from openai import OpenAI
from openai_utils import OPENAI_API_KEY
from datasets.data_helpers import coco_object_categories #will import more later with more datasets
DATASET_NAME_TO_CLASSNAMES = {'COCO2014_partial' : coco_object_categories} #will add more entries later with more datasets


DEBUG = False
DEBUG_NUM_CLASSES = 5
VERBOSE = True
OUT_BASE = '../vislang-domain-exploration-data/dualcoopstarstar-data/llm_generations/classname2list'
LLM_LIMIT = 20


#can generate anything that maps classname ==> classname_list


def query_llm(prompts):
    client = OpenAI(api_key=OPENAI_API_KEY)
    if DEBUG:
        print('prompts:')
        print(prompts)

    print('querying LLM...')
    print('(%d prompts)'%(len(prompts)))
    if len(prompts) <= LLM_LIMIT:
        completion = client.completions.create(model='text-davinci-003', prompt=prompts, temperature=0., max_tokens=300)
        outputs = [c.text for c in completion.choices]
    else:
        cur_t = 0
        outputs = []
        while cur_t < len(prompts):
            cur_prompts = prompts[cur_t:min(cur_t+LLM_LIMIT,len(prompts))]
            completion = client.completions.create(model='text-davinci-003', prompt=cur_prompts, temperature=0., max_tokens=300)
            cur_outputs = [c.text for c in completion.choices]
            outputs.extend(cur_outputs)
            cur_t += LLM_LIMIT

    print('done querying LLM!')
    assert(len(outputs) == len(prompts))
    return outputs


def load_dataset_cfg(dataset_cfg_filename):
    dataset_cfg = CN()
    dataset_cfg.DATASET = CN()
    dataset_cfg.DATASET.NAME = ''

    dataset_cfg.merge_from_file(dataset_cfg_filename)
    dataset_cfg.freeze()

    return dataset_cfg


def load_llm_cfg(llm_cfg_filename):
    llm_cfg = CN()
    llm_cfg.name = ''
    llm_cfg.is_two_stage = 0 #if 0 then use llm_cfg.one_stage, else use llm_cfg.two_stage

    #one-stage algos
    llm_cfg.one_stage = CN()
    llm_cfg.one_stage.prompt_template = ''
    llm_cfg.one_stage.num_to_generate = 0
    llm_cfg.one_stage.num_other_classes_to_use = -1 #<= 0 means we don't use other classes
    llm_cfg.one_stage.other_classes_delimiter = ''
    llm_cfg.one_stage.word_limit = -1 #<= 0 means no word limit
    llm_cfg.one_stage.include_orig_classname = 1
    llm_cfg.one_stage.output_template = ''

    #two-stage algos
    llm_cfg.two_stage = CN()
    llm_cfg.two_stage.first_prompt_template = ''
    llm_cfg.two_stage.first_num_other_classes_to_use = -1
    llm_cfg.two_stage.first_other_classes_delimiter = ''
    llm_cfg.two_stage.first_num_to_generate = -1
    llm_cfg.two_stage.total_num_to_generate = 0
    llm_cfg.two_stage.second_prompt_template = ''
    llm_cfg.two_stage.second_min_num_per_primary = 2 #really a safety factor
    llm_cfg.two_stage.second_num_other_classes_to_use = -1
    llm_cfg.two_stage.second_other_classes_delimiter = ''
    llm_cfg.two_stage.second_word_limit = -1
    llm_cfg.two_stage.include_orig_classname = 1
    llm_cfg.two_stage.output_template = ''

    llm_cfg.merge_from_file(llm_cfg_filename)
    llm_cfg.freeze()

    assert(llm_cfg.name == os.path.splitext(os.path.basename(llm_cfg_filename))[0])

    return llm_cfg


def get_out_filename(llm_cfg, dataset_cfg):
    return get_out_filename_helper(llm_cfg.name, dataset_cfg.DATASET.NAME)


def get_out_filename_helper(llm_name, dataset_name, create_dirs=True):
    out_filename = os.path.join(OUT_BASE, llm_name, dataset_name + '.pkl')
    if create_dirs:
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)

    return out_filename


def get_classnames(dataset_cfg):
    return DATASET_NAME_TO_CLASSNAMES[dataset_cfg.DATASET.NAME]


#assumes numbered list with maybe some extraneous newlines
def postprocess_common(output, lenient=False):
    orig_output = output
    outputs = output.split('\n')
    outputs = [output for output in outputs if len(output.strip()) > 0]
    assert(len(outputs) > 0)
    outputs_ss = [output.split('. ') for output in outputs]
    if not all([len(output_ss) >= 2 for output_ss in outputs_ss]):
        print(orig_output)
        assert(lenient)
        special_ret = []
        for output_ss in outputs_ss:
            if len(output_ss) >= 2:
                assert(output_ss[0].isnumeric())
                special_ret.append('. '.join(output_ss[1:]).strip())
            else:
                assert(len(output_ss[0].split(' and "')) == 2)
                special_ret.append(output_ss[0].split('and "')[1].split('"')[0])

        print(special_ret)
        return special_ret

    assert(all([output_ss[0].isnumeric() for output_ss in outputs_ss]))
    outputs = ['. '.join(output_ss[1:]).strip() for output_ss in outputs_ss]
    return outputs


def postprocess_one_stage(output, classname, llm_cfg):
    llm_cfg = llm_cfg.one_stage
    outputs = postprocess_common(output)
    outputs = [output.replace('.', '').replace('?', '').replace('\n', '').lower() for output in outputs]
    if len(outputs) > llm_cfg.num_to_generate:
        outputs = random.sample(outputs, llm_cfg.num_to_generate)
    elif len(outputs) < llm_cfg.num_to_generate:
        if VERBOSE:
            print('Too few outputs generated for "%s" (%d), duplicate...'%(classname, len(outputs)))
        extra_outputs = random.choices(outputs, k=llm_cfg.num_to_generate-len(outputs))
        outputs.extend(extra_outputs)

    if llm_cfg.include_orig_classname:
        outputs.append(classname)

    if llm_cfg.output_template != '':
        outputs = [llm_cfg.output_template.format(classname=classname, output=output) for output in outputs]

    return outputs


def postprocess_two_stage_first(output, llm_cfg):
    return postprocess_common(output)


def postprocess_two_stage_second(output, llm_cfg, lenient=False):
    outputs = postprocess_common(output, lenient=lenient)
    outputs = [output.replace('.', '').replace('?', '').replace('\n', '').lower() for output in outputs]
    return outputs


def flatten_and_uniquify_2D_list(outputs_list):
    flat_outputs = []
    for outputs in outputs_list:
        flat_outputs.extend(outputs)

    return sorted(set(flat_outputs))


#pare down total length of outputs_list until it's total_num_to_generate
#uses del - deal with it!
def fair_pare(outputs_list, total_num_to_generate):
    while len(flatten_and_uniquify_2D_list(outputs_list)) > total_num_to_generate:
        max_len = max([len(outputs) for outputs in outputs_list])
        argmax_indices = [i for i, outputs in enumerate(outputs_list) if len(outputs) == max_len]
        i = random.choice(argmax_indices)
        j = random.choice(range(max_len))
        del outputs_list[i][j]

    ret = flatten_and_uniquify_2D_list(outputs_list)
    assert(len(ret) == total_num_to_generate)

    return ret


#this will do rightsizing, adding in orig classname if desired, and applying output template if desired
def finalize_two_stage_outputs(outputs_list, classname, llm_cfg):
    llm_cfg = llm_cfg.two_stage
    all_outputs = flatten_and_uniquify_2D_list(outputs_list)
    num_generated = len(all_outputs)
    if num_generated <= llm_cfg.total_num_to_generate:
        if num_generated < llm_cfg.total_num_to_generate:
            if VERBOSE:
                print('Too few outputs generated for "%s" (%d), duplicate...'%(classname, num_generated))
                extra_outputs = random.choices(all_outputs, k=llm_cfg.total_num_to_generate-num_generated)
                all_outputs.extend(extra_outputs)
        else:
            assert(num_generated == llm_cfg.total_num_to_generate)
    else:
        assert(num_generated > llm_cfg.total_num_to_generate)
        all_outputs = fair_pare(outputs_list, llm_cfg.total_num_to_generate)

    if llm_cfg.include_orig_classname:
        all_outputs.append(classname)

    if llm_cfg.output_template != '':
        all_outputs = [llm_cfg.output_template.format(classname=classname, output=output) for output in all_outputs]

    return all_outputs


def make_prompt_one_stage(classname, all_classnames, llm_cfg):
    llm_cfg = llm_cfg.one_stage
    other_classnames_str = None
    if llm_cfg.num_other_classes_to_use > 0:
        other_classnames = random.sample([c for c in all_classnames if c != classname], llm_cfg.num_other_classes_to_use)
        other_classnames_str = llm_cfg.other_classes_delimiter.join(other_classnames)

    prompt = llm_cfg.prompt_template.format(classname = classname, num_to_generate = llm_cfg.num_to_generate, word_limit = llm_cfg.word_limit, other_classanmes_str = other_classnames_str)
    return prompt


def make_prompt_two_stage_first(classname, all_classnames, llm_cfg):
    llm_cfg = llm_cfg.two_stage
    other_classnames_str = None
    if llm_cfg.first_num_other_classes_to_use > 0:
        other_classnames = random.sample([c for c in all_classnames if c != classname], llm_cfg.first_num_other_classes_to_use)
        other_classnames_str = llm_cfg.first_other_classes_delimiter.join(other_classnames)

    prompt = llm_cfg.first_prompt_template.format(classname = classname, num_to_generate = llm_cfg.first_num_to_generate, other_classanmes_str = other_classnames_str)
    return prompt


def make_prompt_two_stage_second(classname, primary_output, num_to_generate, llm_cfg):
    llm_cfg = llm_cfg.two_stage
    other_classnames_str = None
    if llm_cfg.second_num_other_classes_to_use > 0:
        other_classnames = random.sample([c for c in all_classnames if c != classname], llm_cfg.second_num_other_classes_to_use)
        other_classnames_str = llm_cfg.second_other_classes_delimiter.join(other_classnames)

    prompt = llm_cfg.second_prompt_template.format(classname = classname, primary_output = primary_output, num_to_generate = num_to_generate, other_classnames_str = other_classnames_str, word_limit = llm_cfg.second_word_limit)
    return prompt


def one_stage_workflow(classnames, llm_cfg):
    assert(not llm_cfg.is_two_stage)
    prompts = [make_prompt_one_stage(classname, classnames, llm_cfg) for classname in classnames]
    outputs = query_llm(prompts)
    classname_lists = [postprocess_one_stage(output, classname, llm_cfg) for output, classname in zip(outputs, classnames)]
    classname2list = {classname : classname_list for classname, classname_list in zip(classnames, classname_lists)}
    return classname2list


def two_stage_workflow(classnames, llm_cfg):
    assert(llm_cfg.is_two_stage)
    first_stage_prompts = [make_prompt_two_stage_first(classname, classnames, llm_cfg) for classname in classnames]
    first_stage_outputs = query_llm(first_stage_prompts)
    first_stage_outputs_list = [postprocess_two_stage_first(output, llm_cfg) for output in first_stage_outputs]
    classname2indices = {}
    all_second_stage_prompts = []
    assert(len(classnames) == len(first_stage_outputs_list))
    for classname, first_stage_outputs in zip(classnames, first_stage_outputs_list):
        assert(classname not in classname2indices)
        classname2indices[classname] = []
        num_to_generate = int(math.ceil(llm_cfg.two_stage.total_num_to_generate / len(first_stage_outputs)))
        num_to_generate = max(num_to_generate, llm_cfg.two_stage.second_min_num_per_primary)
        for primary_output in first_stage_outputs:
            prompt = make_prompt_two_stage_second(classname, primary_output, num_to_generate, llm_cfg)
            idx = len(all_second_stage_prompts)
            classname2indices[classname].append(idx)
            all_second_stage_prompts.append(prompt)

    all_second_stage_outputs = query_llm(all_second_stage_prompts)
    classname2list = {}
    for classname in classnames:
        second_stage_outputs = [all_second_stage_outputs[idx] for idx in classname2indices[classname]]
        outputs_list = [postprocess_two_stage_second(output, llm_cfg, lenient=True) for output in second_stage_outputs]
        final_outputs = finalize_two_stage_outputs(outputs_list, classname, llm_cfg)
        classname2list[classname] = final_outputs

    return classname2list


def generate_classname2list(llm_cfg_filename, dataset_cfg_filename):
    random.seed(0)
    llm_cfg = load_llm_cfg(llm_cfg_filename)
    dataset_cfg = load_dataset_cfg(dataset_cfg_filename)
    out_filename = get_out_filename(llm_cfg, dataset_cfg)
    classnames = get_classnames(dataset_cfg)

    #yes, this means that other_classnames would get impacted by this
    if DEBUG:
        classnames = random.sample(classnames, DEBUG_NUM_CLASSES)

    if llm_cfg.is_two_stage:
        classname2list = two_stage_workflow(classnames, llm_cfg)
    else:
        classname2list = one_stage_workflow(classnames, llm_cfg)

    if DEBUG:
        import pdb
        pdb.set_trace()

    with open(out_filename, 'wb') as f:
        pickle.dump(classname2list, f)


def usage():
    print('Usage: python generate_classname2list.py <llm_cfg_filename> <dataset_cfg_filename>')


if __name__ == '__main__':
    generate_classname2list(*(sys.argv[1:]))
