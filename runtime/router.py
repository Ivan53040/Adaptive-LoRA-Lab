"""Transparent, bounded input-pattern routing over the existing adapter contracts."""
import re
import time

from runtime.coding_engine import validate_coding_request

VERSION='rules-v1'
TOOL_PATTERNS={
    'turn_on_flashlight':r'\b(?:turn|switch)\s+on\s+(?:my|the|a)?\s*(?:phone\s+)?(?:flashlight|torch)\b',
    'turn_off_flashlight':r'\b(?:turn|switch)\s+off\s+(?:my|the|a)?\s*(?:phone\s+)?(?:flashlight|torch)\b',
    'open_wifi_settings':r'\b(?:open|show)\s+(?:my|the)?\s*(?:wi[ -]?fi|wireless)\s+settings\b',
    'create_contact':r'\b(?:create|add|save)\s+(?:a|new|a new)?\s*contact\b',
    'create_calendar_event':r'\b(?:create|add|schedule)\s+(?:a|new|a new)?\s*(?:calendar\s+)?(?:event|appointment|meeting)\b',
    'send_email':r'\b(?:send|compose)\s+(?:an?|the)?\s*(?:email|e-mail)\b',
    'show_map':r'\b(?:show|open|display)\s+(?:me\s+)?(?:a|the)?\s*map\b',
}
ACTION_PREFIX=r'^(?:(?:please|kindly)\s+|(?:could|can|would)\s+you\s+(?:please\s+)?|i\s+(?:want|need)\s+you\s+to\s+)*(?:turn|switch|open|show|display|create|add|save|schedule|send|compose)\b'
CODE_INTENT=r'\b(?:write|implement|generate|complete|fix|refactor|debug)\b.{0,70}\b(?:python|code|function|algorithm|javascript|java|c\+\+)\b'

def route_request(prompt,allow_experimental_coding=False):
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>4000:
        raise ValueError('Enter a request containing1–4000 characters.')
    if type(allow_experimental_coding) is not bool:
        raise ValueError('Experimental Coding selection must be a boolean.')
    start=time.perf_counter(); text=prompt.strip()
    def decision(domain,adapter,reason,tools=()):
        return {'version':VERSION,'domain':domain,'adapter':adapter,'reason_key':reason,
                'matched_tools':list(tools),'experimental':adapter=='coding',
                'routing_ms':(time.perf_counter()-start)*1000}
    # Code structure takes priority over words inside its docstring.
    if re.match(r'^(?:from\s|import\s|def\s)',text):
        try: validate_coding_request({'prefix':prompt})
        except ValueError: return decision('clarify',None,'routeInvalidPrefix')
        return decision('python','coding' if allow_experimental_coding else 'base',
                        'routeExperimentalCode' if allow_experimental_coding else 'routeBaseCode')
    unquoted=re.sub(r'"[^"\n]*"|\x27[^\x27\n]*\x27','',text).lower()
    code_intent=bool(re.search(CODE_INTENT,unquoted))
    tools=[name for name,pattern in TOOL_PATTERNS.items() if re.search(pattern,unquoted)]
    action=bool(re.search(ACTION_PREFIX,unquoted))
    if action and tools and code_intent:
        return decision('clarify',None,'routeMixed')
    if re.search(r"\b(?:do not|don't|never|avoid)\b",unquoted) and tools:
        return decision('clarify',None,'routeNegated')
    if tools and action:
        return decision('tools','tool','routeTools',tools)
    if code_intent:
        return decision('clarify',None,'routeNeedsPrefix')
    if re.match(r'^(?:please\s+)?(?:delete|remove|erase|book|cancel|turn|switch)\b',unquoted):
        return decision('clarify',None,'routeUnsupported')
    return decision('general','base','routeGeneral')
