import { BASE_PATH } from "@/lib/site";

const SCRIPT = `
(function () {
  var l = (navigator.language || "").toLowerCase();
  var lang = l.indexOf("zh-cn") === 0 || l.indexOf("zh-sg") === 0 || l === "zh-hans" ? "zh-CN"
    : l.indexOf("zh") === 0 ? "zh-TW" : l.indexOf("en") === 0 ? "en" : "zh-TW";
  try { var s = localStorage.getItem("yunshu-docs-lang"); if (s) lang = s; } catch (e) {}
  location.replace(${JSON.stringify(BASE_PATH)} + "/" + lang + "/");
})();
`;

export default function Page() {
  return (
    <>
      <script dangerouslySetInnerHTML={{ __html: SCRIPT }} />
      <noscript>
        <p>
          <a href={`${BASE_PATH}/zh-TW/`}>繁體中文</a> · <a href={`${BASE_PATH}/en/`}>English</a> ·{" "}
          <a href={`${BASE_PATH}/zh-CN/`}>简体中文</a>
        </p>
      </noscript>
    </>
  );
}
