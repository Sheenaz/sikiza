import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { checkIn, flushPending, type TriageResponse } from './src/api';
import { loadLexicon } from './src/safetyValve';
import { colors } from './src/theme';

const COPY = {
  en: {
    title: 'How are you, really?',
    subtitle: 'Write in English, Kiswahili or Sheng — whatever feels natural.',
    placeholder: 'e.g. exams zimeniweza manze…',
    send: 'Check in',
    another: 'New check-in',
    offline: "You're offline. This response comes from the app, not the full service.",
    call: 'Call',
    disclaimer:
      'This is digital first aid, not a diagnosis. If you are in danger, call 1199 or go to the nearest hospital.',
  },
  sw: {
    title: 'Hali yako ikoje kweli?',
    subtitle: 'Andika kwa Kiingereza, Kiswahili au Sheng — vile unavyopenda.',
    placeholder: 'mfano: niko down sana leo…',
    send: 'Tuma',
    another: 'Andika tena',
    offline: 'Huna mtandao. Jibu hili linatoka kwa app, si huduma kamili.',
    call: 'Piga',
    disclaimer:
      'Hii ni huduma ya kwanza ya kidijitali, si uchunguzi wa daktari. Ukiwa hatarini, piga 1199 au nenda hospitali.',
  },
};

export default function App() {
  const [lang, setLang] = useState<'en' | 'sw'>('en');
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TriageResponse | null>(null);
  const t = COPY[lang];

  useEffect(() => {
    loadLexicon();
    flushPending();
  }, []);

  async function submit() {
    if (!text.trim() || busy) return;
    setBusy(true);
    try {
      setResult(await checkIn(text.trim(), lang));
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setResult(null);
    setText('');
  }

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <View style={styles.header}>
          <Text style={styles.brand}>Sikiza</Text>
          <View style={styles.langToggle}>
            {(['en', 'sw'] as const).map((l) => (
              <Pressable
                key={l}
                onPress={() => setLang(l)}
                style={[styles.langBtn, lang === l && styles.langBtnOn]}
                accessibilityRole="button"
                accessibilityState={{ selected: lang === l }}
              >
                <Text style={[styles.langText, lang === l && styles.langTextOn]}>{l.toUpperCase()}</Text>
              </Pressable>
            ))}
          </View>
        </View>

        {!result ? (
          <View>
            <Text style={styles.title}>{t.title}</Text>
            <Text style={styles.subtitle}>{t.subtitle}</Text>
            <TextInput
              style={styles.input}
              multiline
              value={text}
              onChangeText={setText}
              placeholder={t.placeholder}
              placeholderTextColor={colors.muted}
              maxLength={2000}
              textAlignVertical="top"
              accessibilityLabel={t.title}
            />
            <Pressable
              style={[styles.primary, (!text.trim() || busy) && styles.disabled]}
              onPress={submit}
              disabled={!text.trim() || busy}
              accessibilityRole="button"
            >
              {busy ? <ActivityIndicator color="#fff" /> : <Text style={styles.primaryText}>{t.send}</Text>}
            </Pressable>
          </View>
        ) : (
          <View>
            {result.offline && (
              <View style={styles.banner}>
                <Text style={styles.bannerText}>{t.offline}</Text>
              </View>
            )}
            <View style={[styles.card, result.crisis && styles.crisisCard]}>
              <Text style={[styles.cardText, result.crisis && styles.crisisText]}>
                {result.responseText}
              </Text>
              {result.hotlines.map((h) => (
                <Pressable
                  key={h.number}
                  style={styles.callBtn}
                  onPress={() => Linking.openURL(`tel:${h.number}`)}
                  accessibilityRole="button"
                >
                  <Text style={styles.callText}>
                    {t.call} {h.name} · {h.number}
                  </Text>
                </Pressable>
              ))}
            </View>
            <Pressable style={styles.secondary} onPress={reset} accessibilityRole="button">
              <Text style={styles.secondaryText}>{t.another}</Text>
            </Pressable>
            {__DEV__ && (
              <Text style={styles.debug}>
                {result.roundTripMs} ms round trip · {result.responseId}
                {result.category ? ` · ${result.category}` : ''}
              </Text>
            )}
          </View>
        )}
        <Text style={styles.disclaimer}>{t.disclaimer}</Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: 20, paddingTop: 64, paddingBottom: 40 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 },
  brand: { fontSize: 15, fontWeight: '700', color: colors.primary, letterSpacing: 0.3 },
  langToggle: { flexDirection: 'row', backgroundColor: colors.border, borderRadius: 999, padding: 3 },
  langBtn: { paddingHorizontal: 14, paddingVertical: 6, borderRadius: 999 },
  langBtnOn: { backgroundColor: colors.surface },
  langText: { fontSize: 13, fontWeight: '600', color: colors.muted },
  langTextOn: { color: colors.text },
  title: { fontSize: 28, fontWeight: '800', color: colors.text, marginBottom: 8 },
  subtitle: { fontSize: 15, color: colors.muted, marginBottom: 20, lineHeight: 21 },
  input: {
    minHeight: 150,
    backgroundColor: colors.surface,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 16,
    fontSize: 16,
    color: colors.text,
    marginBottom: 16,
  },
  primary: { backgroundColor: colors.primary, borderRadius: 999, paddingVertical: 16, alignItems: 'center' },
  primaryText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  disabled: { opacity: 0.5 },
  banner: { backgroundColor: colors.warnSoft, borderRadius: 12, padding: 12, marginBottom: 12 },
  bannerText: { color: colors.text, fontSize: 13 },
  card: {
    backgroundColor: colors.primarySoft,
    borderRadius: 20,
    padding: 20,
    marginBottom: 16,
  },
  crisisCard: { backgroundColor: colors.crisisSoft, borderWidth: 1, borderColor: colors.crisis },
  cardText: { fontSize: 16, lineHeight: 24, color: colors.text },
  crisisText: { color: colors.text, fontWeight: '500' },
  callBtn: {
    marginTop: 12,
    backgroundColor: colors.crisis,
    borderRadius: 999,
    paddingVertical: 13,
    alignItems: 'center',
  },
  callText: { color: '#fff', fontWeight: '700', fontSize: 15 },
  secondary: {
    borderRadius: 999,
    paddingVertical: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.primary,
  },
  secondaryText: { color: colors.primary, fontWeight: '700', fontSize: 15 },
  debug: { marginTop: 10, fontSize: 12, color: colors.muted, textAlign: 'center' },
  disclaimer: { marginTop: 32, fontSize: 12, color: colors.muted, lineHeight: 17, textAlign: 'center' },
});
