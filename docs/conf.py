import automl_sphinx_theme

from instance_dac import copyright, author, version, name


options = {
    "copyright": copyright,
    "author": author,
    "version": version,
    "name": name,
    "html_theme_options": {
        "github_url": "https://github.com/automl/Instance-DAC",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    }
}

automl_sphinx_theme.set_options(globals(), options)