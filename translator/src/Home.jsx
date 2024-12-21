import React, { useState } from "react";
import "./Home.css";

function Home() {
    const [inputText, setInputText] = useState("");
    const [translation, setTranslation] = useState("");
    const [direction, setDirection] = useState("English to Tagalog");

    const handleTranslate = () => {
        if (inputText.trim() === "") {
            setTranslation("Please enter text to translate.");
            return;
        }
        const result =
            direction === "English to Tagalog"
                ? `Translated to Tagalog: ${inputText}`
                : `Translated to English: ${inputText}`;
        setTranslation(result);
    };

    return (
        <div className="home-container">
            <h1>English to Tagalog Translator</h1>

            <div className="home-content">
                <div className="button-group">
                    <button
                        onClick={() => setDirection("English to Tagalog")}
                        className={
                            direction === "English to Tagalog"
                                ? "active-button"
                                : ""
                        }
                    >
                        English to Tagalog
                    </button>
                    <button
                        onClick={() => setDirection("Tagalog to English")}
                        className={
                            direction === "Tagalog to English"
                                ? "active-button"
                                : ""
                        }
                    >
                        Tagalog to English
                    </button>
                </div>

                <input
                    type="text"
                    placeholder="Enter text here..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    className="input-box"
                />
                <button onClick={handleTranslate} className="translate-button">
                    Translate
                </button>

                <div className="translate-display">
                    <strong>
                        {translation || "Your translation will appear here."}
                    </strong>
                </div>
            </div>
        </div>
    );
}

export default Home;
