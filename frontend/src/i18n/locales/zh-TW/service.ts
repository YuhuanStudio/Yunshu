const service = {
  title: "服務",
  description: "引擎作為背景服務（launchd）的狀態，以及重新啟動。",
  refresh: "重新整理",
  loading: "讀取服務狀態…",
  "unavailable.unsupported.title": "此引擎版本沒有服務管理",
  "unavailable.unsupported.description":
    "升級 Yunshu 後即可在此檢視與重新啟動。",
  "unavailable.denied.title": "需要管理員權杖",
  "unavailable.denied.description":
    "請在「引擎連線」輸入服務的 YUNSHU_AUTH_TOKEN 或具管理權限的金鑰。",
  "unavailable.error.title": "無法讀取",
  "unavailable.error.description": "請確認服務仍在運行，稍後再試。",
  "state.managed": "由 launchd 管理",
  "state.other": "launchd 已載入，但目前這個進程不是它啟動的",
  "state.stopped": "已安裝，未運行",
  "state.notInstalled": "尚未安裝為服務",
  "row.status": "狀態",
  "row.pid": "進程 ID",
  "row.uptime": "已運行",
  "row.uptimeHelp": "自此進程啟動起算。",
  "row.version": "版本",
  "row.plist": "plist 檔案",
  "row.log": "日誌檔案",
  "restart.button": "重新啟動",
  "restart.confirmTitle": "重新啟動引擎",
  "restart.confirmBody":
    "會先等待進行中的請求完成，最久 {seconds} 秒，之後強制重啟。重啟期間服務暫時無法連線，已載入的模型需要重新載入。",
  "restart.confirmBodyUnknown":
    "會先等待進行中的請求完成，之後強制重啟。重啟期間服務暫時無法連線，已載入的模型需要重新載入。",
  "restart.confirm": "重新啟動",
  "restart.accepted":
    "已安排重新啟動：正在等待 {count} 個進行中的請求（最久 {seconds} 秒）。完成後控制台會自動重新連線。",
  "restart.failed": "重新啟動沒有成功送出，請稍後再試。",
  "restart.notLaunchd":
    "這個引擎不是由 launchd 服務啟動的，無法自行重啟。請手動重新啟動：",
  "restart.manual": "手動重新啟動的指令",
  "restart.cli": "指令",
  "restart.help": "一鍵重新啟動只在由 launchd 服務管理時可用。",
  "network.title": "網路",
  "network.description":
    "服務監聽的位址（唯讀）。要改變監聽位址，請重新安裝服務。",
  "network.address": "控制台連線的位址",
  "network.addressHelp": "來自目前的服務位址設定。",
  "network.exposure": "可連線範圍",
  "network.exposureHelp":
    "僅本機表示只有這台電腦能連；區域網路表示同網路的其他裝置也能連。",
  "network.loopback": "僅本機",
  "network.lan": "區域網路可連線",
  "network.cmdLocal": "改為僅本機",
  "network.cmdLan": "改為區域網路可連線",
  "network.lanWarning":
    "開放區域網路前，請先設定 YUNSHU_AUTH_TOKEN 或建立 API 金鑰，並在 CORS 只允許需要的來源。",
  "cors.title": "CORS 來源",
  "cors.description": "允許哪些網站從瀏覽器呼叫這個服務。儲存後立即生效。",
  "cors.invalid":
    "請輸入 http:// 或 https:// 開頭的來源，不含路徑，例如 https://app.example.com；或單獨的 *。",
  "cors.mixed": "* 不能和其他來源同時使用。",
  "cors.duplicate": "這個來源已在清單中。",
  "cors.rejected": "伺服器拒絕了這些來源：{list}",
  "cors.forced":
    "目前由{source}設定，優先於設定檔；這裡的修改會儲存，但不會改變運行中的值。",
  "cors.source.env": "環境變數 YUNSHU_CORS_ORIGINS",
  "cors.source.cli": "命令列參數",
  "cors.wildcardTitle": "任何網站都能呼叫這個服務",
  "cors.wildcardBody":
    "* 讓任何網站的瀏覽器頁面都能向此服務發請求，且不會帶憑證。除非服務只在本機可連，否則請改用明確的來源。",
  "cors.thisAllowed": "目前的控制台來源 {origin} 已被允許。",
  "cors.thisBlocked":
    "目前的控制台來源 {origin} 不在清單中，其他瀏覽器頁面會被擋下。",
  "cors.listAria": "允許的來源",
  "cors.remove": "移除 {origin}",
  "cors.none": "清單是空的",
  "cors.add": "新增來源",
  "cors.addButton": "新增",
  "cors.confirmAny": "我了解 * 會讓任何網站呼叫此服務",
  "cors.confirmAnyNeeded": "使用 * 前需要先確認",
  "cors.save": "儲存",
  "cors.reset": "還原預設",
  "cors.alreadyDefault": "目前已是預設值",
  "cors.saved": "已儲存",
  "cors.help": "預設：{default}。",
};
export default service;
