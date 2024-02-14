WELCOME = """
Bonjour, je suis un chatbot concu pour aider à trouver des informations dans votre base documentaire.
"""

ASK_FOLDER = """
Afin de vous assister, pourriez-vous m'indiquer dans quel dossier local se trouvent vos documents ?
"""

FOLDER_NOT_EXIST = """Je suis désolé, le dossier n'existe pas."""

FOLDER_N_FIRST_FILES = """J'ai identifié votre dossier. Voici les {} premiers fichiers : \n\n{}"""

LOADING_DONE = """J'ai terminé de charger les documents. Vous pouvez commencer à poser vos questions."""

TEMPLATE_RESPONSE = """
J'ai trouvé {} résultats pour votre recherche : \n\n{}
"""

TEMPLATE_NODE = "{}. (score: {:.2f})\n{}\n\n"

SYSTEM_PROMPT_MISTRAL = """<s>[INST] You are a helpful assistant.
 Your task is to answer the user query using the provided context. [/INST]"""

QUERY_WRAPPER_PROMPT_MISTRAL = "[INST] {query_str} [/INST]"
