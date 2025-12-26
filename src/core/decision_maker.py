"""
DecisionMaker
-------------
Integrates multiple analysts and produces actionable investment decisions.
"""
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

class DecisionMaker:
    def __init__(self, local_dir):
        self.local_dir = local_dir

        warnings.filterwarnings(
            "ignore",
            message="To copy construct from a tensor.*",
            category=UserWarning,
        )
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir,
            local_files_only=True
        )

        print("Loading model (this is slow only ONCE)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="mps",
            low_cpu_mem_usage=True
        )

    def classify(self, disclosure: str):
        messages = [
    {
        "role": "system",
        "content": (
            "You are a financial disclosure classifier.\n"
            "Rules (MANDATORY):\n"
            "- Form 144 and insider selling disclosures MUST be classified as NEUTRAL_OR_UNCLEAR.\n"
            "- Employee compensation (RSU, stock grants) is NOT a business contract.\n"
            "- Do NOT infer revenue impact unless explicitly stated.\n\n"
            "Output format:\n"
            "Label: <POSITIVE_CONTRACT | PROFIT_INCREASE | NEUTRAL_OR_UNCLEAR>\n"
            "Reason: <brief factual explanation>\n\n"
            "Important:\n"
            "- Decide the Label first.\n"
            "- The Reason must justify the Label, not reinterpret the disclosure.\n"
            "- If unsure, choose NEUTRAL_OR_UNCLEAR."
        )
    },
    {
        "role": "user",
        "content": (
            "Background knowledge:\n"
            "- Form 144 is a notice of proposed sale of securities by insiders.\n"
            "- It does NOT represent a business contract or profit increase.\n"
            "- Such disclosures are usually informational or neutral."
        )
    },
    {
        "role": "user",
        "content": (
            "Disclosure:\n"
            f"{disclosure}"
        )
    }
]


        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                temperature=None,
                top_p=0.85,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


        messages = [
            {"role": "system", "content": "You are an AI that helps with reasonable investment decisions."},
            {"role": "user", "content": """Classify the disclosure as one of:
        - POSITIVE_CONTRACT
        - PROFIT_INCREASE
        - NEUTRAL_OR_UNCLEAR
        
        IMPORTANT RULES:
        - Form 144, insider sales, or proposed sale notices are NOT positive contracts.
        - Only classify POSITIVE_CONTRACT if the disclosure explicitly states
        a new business contract that increases revenue.
        - Insider selling notices must be classified as NEUTRAL_OR_UNCLEAR.
        - Answer with only one label and a short reason."""}
        ]




# poetry run python3 src/core/decision_maker.py

if __name__ == "__main__":
    dm = DecisionMaker("models/qwen2.5-1.5b-instruct")
    # dm2 = DecisionMaker("models/phi-3-medium.Q4_K_M.gguf") -> phi-3 llama based, no AutoTokenizer
    
    # dm.download_model_in_cache()
    result = dm.classify("""Form 144 Filer Information
        UNITED STATES
        SECURITIES AND EXCHANGE COMMISSION
        Washington, D.C. 20549
        Form 144
        NOTICE OF PROPOSED SALE OF SECURITIES
        PURSUANT TO RULE 144 UNDER THE SECURITIES ACT OF 1933
        Form 144
        144: Issuer Information
        Name of Issuer Apple Inc.
        SEC File Number 001-36743
        Address of Issuer One Apple Park Way
        Cupertino
        CALIFORNIA
        95014
        Phone 4089961010
        Name of Person for Whose Account the
        Securities are To Be Sold KEVAN PAREKH
        See the definition of "person" in paragraph (a) of Rule 144. Information is to be given not only as to the person for whose account
        the securities are to be sold but also as to all other persons included in that definition. In addition, information shall be given as to
        sales by all persons whose sales are required by paragraph (e) of Rule 144 to be aggregated with sales for the account of the
        person filing this notice.
        Relationship to Issuer 1. Officer
        144: Securities Information
        Record Title of the Class of
        Securities To Be Sold
        Name and Address of the
        Broker
        Number of
        Shares or Other
        Units To Be Sold
        Aggregate
        Market Value
        Number of
        Shares or Other
        Units
        Outstanding
        Approximate
        Date of Sale
        Name the
        Securities
        Exchange
        #1 Common
        Morgan Stanley Smith
        Barney LLC Executive
        Financial Services
        1 New York Plaza
        8th Floor
        New York
        NEW YORK
        10004
        4,199 $1,046,978.66 14,840,390,000 10/16/2025 NASDAQ
        144: Securities To Be Sold
        Furnish the following information with respect to the acquisition of the securities to be sold and with respect to the payment of all
        or any part of the purchase price or other consideration therefor:
        Record Title of the Class Date you
        Acquired
        Nature of
        Acquisition
        Transaction
        Name of Person
        from Whom
        Acquired
        Is
        this
        a
        Gift
        ?
        Date
        Donor
        Acquired
        Amount of
        Securities
        Acquired
        Date of
        Payment
        Nature of
        Payment *
        #1 Common 10/15/2025 Restricted Stock
        Units Issuer ☐ — 4,199 10/15/2025 N/A
        * If the securities were purchased and full payment therefor was not made in cash at the time of purchase, explain in the table or in
        a note thereto the nature of the consideration given. If the consideration consisted of any note or other obligation, or if payment
        was made in installments describe the arrangement and state when the note or other obligation was discharged in full or the last
        installment paid.
        144: Securities Sold During The Past 3 Months
        Nothing to Report ☒
        144: Remarks and Signature
        Remarks
        Date of Notice 10/16/2025
        Date of Plan Adoption or Giving of
        Instruction, If Relying on Rule 10b5-1 1. 11/26/2024
        ATTENTION:
        The person for whose account the securities to which this notice relates are to be sold hereby represents by signing this notice that
        he does not know any material adverse information in regard to the current and prospective operations of the Issuer of the
        securities to be sold which has not been publicly disclosed. If such person has adopted a written trading plan or given trading
        instructions to satisfy Rule 10b5-1 under the Exchange Act, by signing the form and indicating the date that the plan was adopted
        or the instruction given, that person makes such representation as of the plan adoption or instruction date.
        Signature /s/ Kevan Parekh
        ATTENTION: Intentional misstatements or omission of facts constitute Federal Criminal Violations (See 18 U.S.C. 1001)""")
    print("Decision Maker ", dm.local_dir)
    print(result)
