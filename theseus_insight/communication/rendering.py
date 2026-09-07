"""Pure newsletter HTML rendering shared by email and local previews."""
import markdown2


def render_newsletter_html(content):
    """
    Create enhanced HTML content with proper CSS styling to prevent title formatting issues.

    Args:
        content (str): The markdown content

    Returns:
        str: Enhanced HTML content
    """
    # Pre-process content to clean up any problematic title formatting
    import re

    # Clean up titles in the markdown content first
    def clean_title_line(match):
        title_text = match.group(1)
        # Remove any newlines or extra whitespace within the title
        cleaned_title = re.sub(r'\s+', ' ', title_text).strip()
        return f"## {cleaned_title}"

    # Pattern to match ## title lines and clean them
    title_pattern = r'^## (.+?)$'
    cleaned_content = re.sub(title_pattern, clean_title_line, content, flags=re.MULTILINE)

    # Convert markdown to HTML
    base_html = markdown2.markdown(cleaned_content)

    # Enhanced CSS specifically for email clients like Gmail
    email_css = """
    <style type="text/css">
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: bold !important;
            line-height: 1.3 !important;
            /* Prevent title breaking across lines from losing formatting */
            display: block !important;
            word-wrap: break-word !important;
            /* Force consistent bold formatting across line breaks */
            font-weight: 700 !important;
            /* Prevent unwanted line breaks within titles */
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        h2 {
            font-size: 1.5em !important;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            /* Allow line breaks but maintain formatting */
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            /* Ensure the entire title block maintains formatting */
            overflow-wrap: break-word !important;
        }
        /* Force Gmail to respect our styling */
        .title-block {
            font-weight: bold !important;
            font-size: 1.5em !important;
            color: #2c3e50 !important;
            border-bottom: 2px solid #3498db !important;
            padding-bottom: 10px !important;
            margin-top: 30px !important;
            margin-bottom: 15px !important;
            display: block !important;
            line-height: 1.3 !important;
            /* Better handling of long titles */
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            hyphens: auto !important;
            /* Ensure consistent formatting across all parts of the title */
            font-family: inherit !important;
        }
        /* Additional span styling for title components */
        .title-block span {
            font-weight: bold !important;
            font-size: inherit !important;
            color: inherit !important;
            font-family: inherit !important;
        }
        hr {
            border: 0;
            border-top: 1px solid #dce4eb;
            margin: 32px 0;
        }
        p {
            margin-bottom: 15px;
            line-height: 1.6;
        }
        ul, ol {
            margin-bottom: 15px;
            padding-left: 30px;
        }
        li {
            margin-bottom: 5px;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        /* Specific styles for mobile Gmail apps */
        @media only screen and (max-width: 600px) {
            body {
                padding: 10px;
            }
            h2, .title-block {
                font-size: 1.3em !important;
            }
        }
    </style>
    """

    # Post-process the HTML to add better title formatting
    # Replace h2 tags with enhanced div blocks to ensure consistent formatting
    def replace_h2_with_enhanced(match):
        title_text = match.group(1)
        # Additional cleaning of the title text
        clean_title = re.sub(r'\s+', ' ', title_text).strip()
        # Wrap in spans to ensure formatting consistency
        return f'<div class="title-block"><span>{clean_title}</span></div>'

    # Pattern to match h2 tags and their content
    h2_pattern = r'<h2>(.*?)</h2>'
    enhanced_html = re.sub(h2_pattern, replace_h2_with_enhanced, base_html, flags=re.DOTALL)

    # Wrap in proper HTML structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Theseus Insight Newsletter</title>
        {email_css}
    </head>
    <body>
        {enhanced_html}
    </body>
    </html>
    """

    return full_html

