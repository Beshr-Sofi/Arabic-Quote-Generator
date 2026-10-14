from html import escape

import gradio as gr

from src.modeling.pipeline import run_safe

SUPPORT = ("يبدو أنك تمرّ بوقت صعب جدًا، وهذا يستحق أن تتحدث عنه مع شخص تثق به أو مع مختص. "
           "وإذا كنت في خطر فوري فتواصل مع خدمات الطوارئ في بلدك.")


def card(text: str, small: bool = False) -> str:
    """Render a text snippet inside an HTML card styled for Right-to-Left (RTL) Arabic display.

    Escapes HTML entities in the text to prevent XSS/rendering issues and formats font size,
    borders, line-height, and padding.

    Args:
        text (str): The text message or quote to display.
        small (bool, optional): Whether to use smaller font (18px for alerts) or larger (30px for quotes). Defaults to False.

    Returns:
        str: Styled HTML div string.
    """
    size = "18px" if small else "30px"
    return (f'<div style="direction:rtl;text-align:center;font-size:{size};line-height:1.9;'
            f'padding:24px;border-radius:12px;border:1px solid #8884">{escape(text)}</div>')


def make_quote(statement: str) -> str:
    """Process user input from Gradio interface and return formatted HTML output.

    Handles edge cases:
    - Empty input -> prompts user to enter text.
    - Crisis detected -> displays supportive crisis resource message.
    - Rate limit/API error -> notifies user that daily limit was reached.
    - No quote generated -> prompts user to try a different phrasing.
    - Success -> returns the generated Arabic quote inside an HTML card.

    Args:
        statement (str): User statement entered in the text box.

    Returns:
        str: Rendered HTML card representing the response state.
    """
    statement = (statement or "").strip()
    if not statement:
        return card("اكتب جملة أو كلمة أولًا", small=True)

    res = run_safe(statement)
    print(res)  # tags and candidates now show in the terminal instead of the page
    if res.get("crisis"):
        return card(SUPPORT, small=True)
    if res.get("error"):
        return card("وصلنا إلى حدّ الاستخدام المجاني، حاول لاحقًا", small=True)
    if not res.get("arabic"):
        return card("لم أجد اقتباسًا مناسبًا، جرّب صياغة أخرى", small=True)
    return card(res["arabic"])


with gr.Blocks(title="مولّد الاقتباسات") as demo:
    gr.Markdown("# مولّد الاقتباسات\nاكتب جملة أو كلمة، وسأكتب لك اقتباسًا.")
    inp = gr.Textbox(label="جملتك", placeholder="مثال: مررت بأوقات صعبة لكنني لم أستسلم",
                     rtl=True, lines=2)
    btn = gr.Button("اكتب اقتباسًا", variant="primary")
    out = gr.HTML()
    btn.click(make_quote, inp, out)
    inp.submit(make_quote, inp, out)

if __name__ == "__main__":
    demo.launch()