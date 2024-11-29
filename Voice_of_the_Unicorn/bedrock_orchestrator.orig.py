import json
import boto3
import os

TEAMDDBTABLE = os.getenv('TEAM_DDB_TABLE')

# Lambda handler function
def lambda_handler(event, context):
    transcript = create_plain_transcript(event)
    bedrock_output = get_bedrock_results(transcript)
    response = update_DDB(bedrock_output, event['JobName'], TEAMDDBTABLE)
    return response

# A function that converts Transcribe JSON output into plain text transcript. For example -
# AGENT: Hi!
# CUSTOMER: Hello, I am calling about my credit card
def create_plain_transcript(transcript_event):
    print(transcript_event)
    op = ""
    for i in transcript_event['Transcript']:
        op=op+i['ParticipantRole']+": "+i['Content']+"\n"
    return op

# A function that calls Bedrock to get LLM's output. In this case, calling ClaudeV2
def get_bedrock_results(plain_transcript):
    bedrock = boto3.client(service_name='bedrock-runtime')
    prompt_template = """<br>
<br>Human: Answer all the questions below, based on the contents of <transcript></transcript>, as a json object with key value pairs. Use the text before the colon as the key, and the answer as the value.  If you cannot answer the question, reply with 'n/a'. Only return json. Use gender neutral pronouns. Skip the preamble; go straight into the json.
<br>
<br><questions>
<br>Summary: Summarize the transcript in no more than 5 sentences. Were the caller's needs met during the call?
<br>Topic: Topic of the call. Choose from one of these or make one up (unicorn issue, billing issue, cancellation)
<br>Product: What product did the customer call about? (unicorn rental, unicorn warranty, unicorn accessory, unicorn legs)
<br>Resolved: Did the agent resolve the customer's questions? (yes or no) 
<br>Callback: Was this a callback? (yes or no) 
<br>Politeness: Was the agent polite and professional? (yes or no)
<br>Actions: What actions did the Agent take? 
<br></questions> 
<br>
<br><transcript>
<br>{transcript}
<br></transcript>
<br>
<br>Assistant:"""
    prompt_template = prompt_template.replace("<br>","\n")
    body = json.dumps({
        "prompt": prompt_template.format(transcript = plain_transcript),
        "max_tokens_to_sample": 10000,
        "temperature": 0.1,
        "top_p": 0.9,
    })

    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    response_body = json.loads(response.get('body').read())

    '''
    returns a JSON in the following format:
    response = {
        "Summary": <string>,
        "Topic": <string>,
        "Product": <string>,
        "Resolved": <string>,
        "Callback": <string>,
        "Politeness": <string>,
        "Actions": <string>
    }
    '''

    return json.loads(response_body.get('completion'))


def update_DDB(bedrock_op, job_name, table_name):
    #Todo: extract values from bedrock_op corresponding to each variable
    bedrock_op_summary = None
    bedrock_op_topic = None
    bedrock_op_product = None
    bedrock_op_resolved = None
    bedrock_op_callback = None
    bedrock_op_politeness = None
    bedrock_op_actions = None

    #Todo: Init boto3 dynamoDB client in ddb variable
    ddb = None
    response = ddb.put_item(
        TableName=table_name,
        Item={
            'call_analytics_job_name':{'S':job_name},
            'summary': {'S':bedrock_op_summary},
            'topic': {'S':bedrock_op_topic},
            'product': {'S':bedrock_op_product},
            'resolved': {'S':bedrock_op_resolved},
            'callback': {'S':bedrock_op_callback},
            'politeness': {'S':bedrock_op_politeness},
            'actions': {'S':bedrock_op_actions}
            }
        )
    return response

