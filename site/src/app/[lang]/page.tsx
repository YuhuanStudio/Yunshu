import { notFound } from "next/navigation";
import { Home } from "@/components/home";
import { isLang } from "@/lib/i18n";

export default async function HomePage(props: { params: Promise<{ lang: string }> }) {
  const { lang } = await props.params;
  if (!isLang(lang)) notFound();
  return <Home lang={lang} />;
}
