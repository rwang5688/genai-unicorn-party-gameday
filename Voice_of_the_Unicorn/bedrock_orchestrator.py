import json
import boto3
import os

TEAMDDBTABLE = os.getenv('TEAM_DDB_TABLE')

PROMPT_TEMPLATE = """
Generate the characteristics of the call whose transcript is present in <transcript></transcript>? Use gender neutral pronouns.

<transcript>
{transcript}
</transcript>

"""

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "voice_analytics",
                "description": "Get the characteristics of a call between a call center agent and customer, using the transcript as the input. The interactions are for a company that sells unicorn related products.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "Summary": {
                                "type": "string",
                                "description": "Summary of the call in no more than 50 words. Summary is for call center managers or executives, presented in passive voice."
                            },
                            "Topic": {
                                "type": "string",
                                "description": "Topic of the call. Can be from one of these or something else (unicorn issue, billing issue, cancellation)."
                            },
                            "Product": {
                                "type": "string",
                                "description": "What product did the customer call about? (unicorn rental, unicorn warranty, unicorn accessory, unicorn legs)."
                            },
                            "Resolved": {
                                "type": "string",
                                "description": "Did the agent resolve the customer's questions? (yes or no)"
                            },
                            "Callback": {
                                "type": "string",
                                "description": "Was this a callback? (yes or no)"
                            },
                            "Politeness": {
                                "type": "string",
                                "description": "Was the agent polite and professional? (yes or no)"
                            },
                            "Actions": {
                                "type": "string",
                                "description": "What actions did the Agent take? "
                            }
                        },
                        "required": [
                            "Summary",
                            "Topic",
                            "Product",
                            "Resolved",
                            "Callback",
                            "Politeness",
                            "Actions"
                        ]
                    }
                }
            }
        }
    ],
    "toolChoice": {"any":{}}
}



# Lambda handler function
def lambda_handler(event, context):
    transcript = create_plain_transcript(event)
    bedrock_output = get_bedrock_results(transcript)
    response = update_DDB(bedrock_output, event['JobName'], TEAMDDBTABLE)
    return response

def create_plain_transcript(transcript_event):
    """
    A function that converts Transcribe JSON output into plain text transcript. For example -
    # AGENT: Hi!
    # CUSTOMER: Hello, I am calling about my credit card
    """
    op = ""
    for i in transcript_event['Transcript']:
        op=op+i['ParticipantRole']+": "+i['Content']+"\n"
    return op

# A function that calls Bedrock to get LLM's output. In this case, calling Claude 3 Haiku
def get_bedrock_results(plain_transcript):
    bedrock = boto3.client(service_name='bedrock-runtime')

    model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
    messages = [{
        "role": "user",
        "content": [{"text": PROMPT_TEMPLATE.format(transcript = plain_transcript)}]
    }]

    # TODO Fill in the guardrail config values
    guardrail_config = {
                "guardrailIdentifier": "ci02880ajnyp",
                "guardrailVersion": "1",
                "trace": "enabled"
            }
    
    response = bedrock.converse(
        modelId=model_id,
        messages=messages,
        toolConfig=TOOL_CONFIG,
        guardrailConfig=guardrail_config
    )

    return response['output']['message']['content'][0]['toolUse']['input']
    '''
    returns a JSON in the following format:
    {
        "Summary": <string>,
        "Topic": <string>,
        "Product": <string>,
        "Resolved": <string>,
        "Callback": <string>,
        "Politeness": <string>,
        "Actions": <string>
    }
    '''


def update_DDB(bedrock_op, job_name, table_name):
    ddb = boto3.client('dynamodb')
    response = ddb.put_item(
        TableName=table_name,
        Item={
            'call_analytics_job_name':{'S':job_name},
            'summary': {'S':bedrock_op["Summary"]},
            'topic': {'S':bedrock_op["Topic"]},
            'product': {'S':bedrock_op["Product"]},
            'resolved': {'S':bedrock_op["Resolved"]},
            'callback': {'S':bedrock_op["Callback"]},
            'politeness': {'S':bedrock_op["Politeness"]},
            'actions': {'S':bedrock_op["Actions"]}
            }
        )
    return response

