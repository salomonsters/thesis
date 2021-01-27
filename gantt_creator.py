import datetime
from datetime import date, timedelta
import pandas as pd




class Gantt:
    colors = ('orange', 'blue', 'green')
    one_day = timedelta(days=1)

    def __init__(self, start_date, duration_ignores_weekend_days=True):
        self.groups = {}
        self.start_date = start_date
        self.end_date = None
        self.named_objects = {}
        self.duration_ignores_weekend_days = duration_ignores_weekend_days
        self.links = []
        self.links_columns = ['from', 'to', 'link_options']

    def _update_global_end_date(self, end_date):
        if self.end_date is None:
            self.end_date = end_date
        else:
            self.end_date = max(self.end_date, end_date)

    def _get_end_date_and_date_tuples(self, start_date, duration):
        duration_in_days = duration/self.one_day
        date = start_date
        week_days_passed = 0

        date_tuples = []
        date_tuple_start_date = start_date

        # If something takes 5 days, first day should be inclusive
        while week_days_passed < duration_in_days - 1:
            date = date + self.one_day
            if not self.duration_ignores_weekend_days:
                week_days_passed += 1
            else:
                if date.weekday() < 5:
                    week_days_passed += 1

                if date.weekday() == 4:
                    date_tuples.append((date_tuple_start_date, date))
            if date.weekday() == 0 and week_days_passed > 0:
                # Second, third, ... week starts
                date_tuple_start_date = date
        # todo remove extra tuple for DRP1.
        if date > date_tuples[-1][1]:
            date_tuples.append((date_tuple_start_date, date))
        if not self.duration_ignores_weekend_days:
            return date, [(start_date, date)]
        else:
            return date, date_tuples

    def _get_start_date_after(self, end_date):
        if not self.duration_ignores_weekend_days:
            return end_date + self.one_day

        weekday = end_date.weekday()
        if weekday <= 3: # mon-thur
            return end_date + self.one_day
        else:
            return end_date + (7 - weekday) * self.one_day

    def add_group(self, name, text, start_day_after=None, start_date=None, color_number=0, line_after=False):
        if start_day_after is None and start_date is None:
            raise ValueError("Either start_after or start_date should be provided")
        elif len(self.groups) == 0 and start_date is None:
            raise ValueError("First group added should have a start date")
        elif start_day_after is not None:
            if start_date is not None:
                raise ValueError("Cannot provide both start_after and start_date")
            start_date = self._get_start_date_after(self.named_objects[start_day_after]["end_date"])
        self.groups[name] = {
            "text": text,
            "start_date": start_date,
            "end_date": start_date,
            "dates_override": None,
            "color": self.colors[color_number],
            "line_after": line_after,
            "items": []
        }
        self.named_objects[name] = self.groups[name]
        self._update_global_end_date(start_date + self.one_day)

    def add_item(self, text, group_name, duration, start_day_after=None, start_date=None, name=None, link=False, link_options=None):
        if start_day_after is None and start_date is None:
            raise ValueError("Either start_after or start_date should be provided")
        elif start_day_after is not None:
            if start_date is not None:
                raise ValueError("Cannot provide both start_after and start_date")
            if duration == self.one_day:
                start_date = self.named_objects[start_day_after]["end_date"]
            else:
                start_date = self._get_start_date_after(self.named_objects[start_day_after]["end_date"])
            if link:
                if name is None:
                    raise ValueError("Need to provide a name to allow a link")
                self.links.append({'from': start_day_after, 'to': name, 'link_options': link_options})

        dates_override = None
        if duration == self.one_day:
            # Treat 1-day durations as milestones
            end_date = start_date
            dates_override = "milestone"
        else:
            end_date, date_tuples = self._get_end_date_and_date_tuples(start_date, duration)

            if len(date_tuples) > 1:
                dates_override = date_tuples
        item = {
                "name": name,
                "text": text,
                "start_date": start_date,
                "end_date": end_date,
                "dates_override": dates_override
            }
        self.groups[group_name]['items'].append(item)
        self.groups[group_name]['end_date'] = max(end_date, self.groups[group_name]['end_date'])
        if name is not None:
            self.named_objects[name] = item
        self._update_global_end_date(end_date)


    preamble_code = r"""
    \usepackage{color}
    \usepackage{tikz}

    \usepackage{pgfgantt}
    %Used to draw gantt charts, which I will use for the calendar.
    %Let's define some awesome new ganttchart elements:
    \newganttchartelement{orangebar}{
        orangebar/.style={
            inner sep=0pt,
            draw=red!66!black,
            very thick,
            top color=white,
            bottom color=orange!80
        },
        orangebar label font=\slshape,
        orangebar left shift=.1,
        orangebar right shift=-.1
    }

    \newganttchartelement{bluebar}{
        bluebar/.style={
            inner sep=0pt,
            draw=purple!44!black,
            very thick,
            top color=white,
            bottom color=blue!80
        },
        bluebar label font=\slshape,
        bluebar left shift=.1,
        bluebar right shift=-.1
    }

    \newganttchartelement{greenbar}{
        greenbar/.style={
            inner sep=0pt,
            draw=green!50!black,
            very thick,
            top color=white,
            bottom color=green!80
        },
        greenbar label font=\slshape,
        greenbar left shift=.1,
        greenbar right shift=-.1
    }"""


    def to_latex(self):
        links = pd.DataFrame(self.links, columns=self.links_columns)
        project_end_date = self.end_date
        if project_end_date.weekday() < 6:
            project_end_date = project_end_date + (6 - project_end_date.weekday()) * self.one_day
        output = r"""
        \documentclass{standalone}
        \begin{document}
        \begin{landscape}
        \begin{ganttchart}[
            hgrid style/.style={black, dotted},
            vgrid={*5{black,dotted}, *1{white, dotted}, *1{black, dashed}},
            x unit=2.5mm,
            y unit chart=8mm,
            y unit title=12mm,
            time slot format=isodate,
            group label font=\bfseries \Large,
            link/.style={->, thick}]"""
        output += f"{{{self.start_date}}}{{{project_end_date}}}"
        output += r"""
            \gantttitlecalendar{year, month=name, week}\\
        """
        for group_name, group in self.groups.items():
            if group['dates_override'] is None:
                dates = [(group['start_date'], group['end_date'])]
            else:
                dates = group['dates_override']
            for i, date_tuple in enumerate(dates):
                text_string = ""
                if i == 0:
                    text_string = group['text']
                output += f"""\\ganttgroup[group/.append style={{fill={group['color']}}}]{{{text_string}}}{{{date_tuple[0]}}}{{{date_tuple[1]}}}"""
            output += "\\\\ [grid]\n"
            for j, item in enumerate(group['items']):
                if item['dates_override'] is None or item['dates_override'] == 'milestone':
                    item_dates = [(item['start_date'], item['end_date'])]
                else:
                    item_dates = item['dates_override']
                for k, date_tuple in enumerate(item_dates):
                    item_text_string = ""
                    item_name_string = ""
                    linked_string = "linked"
                    if k == 0:
                        item_text_string = item['text']
                        linked_string = ""
                    # only apply name on last item of date tuple for linking purposes
                    if item['name'] is not None:
                        if k == 0:
                            item_name_string = "[name={}]".format(item['name'])
                        elif k == len(item_dates) - 1:
                            item_name_string = "[name={}-last]".format(item['name'])
                            links[links['from'] == item['name']]['from'] = "{}-last".format(item['name'])
                    if item['dates_override'] == 'milestone':
                        output += f"""\\ganttmilestone{item_name_string}{{{item_text_string}}}{{{date_tuple[0]}}}"""
                    else:
                        output += f"""\\gantt{linked_string}{group['color']}bar{item_name_string}{{{item_text_string}}}{{{date_tuple[0]}}}{{{date_tuple[1]}}}"""
                if j == len(group['items']) - 1 and group_name == list(self.groups.keys())[-1]:
                    output += '\n'
                else:
                    output += "\\\\ [grid]\n"

            if group['line_after']:
                output += "\n             \\ganttnewline[thick, black]\n"

            for _, link in links.iterrows():
                source, dest, link_options = link
                link_options_code = ""
                if link_options is not None:
                    link_options_code = "[{}]".format(link_options)
                output += f"\\ganttlink{link_options_code}{{{source}}}{{{dest}}}"

        output += r"""
        \end{ganttchart}
        \end{landscape}
        \end{document}"""
        return output


if __name__ == "__main__":
    project_start_date = date(2021, 1, 25)
    G = Gantt(project_start_date)
    G.add_group(name="DRP", text="Research Phase", color_number=0, start_date=project_start_date)
    G.add_item(text="Clustering", name="DRP-1", group_name="DRP", duration=timedelta(days=7), start_date=project_start_date)
    G.add_item(text="Property extraction", name='DRP-2', group_name="DRP", duration=timedelta(days=6), start_day_after="DRP-1")
    G.add_item(text="Verification", name='DRP-3', group_name="DRP", duration=timedelta(days=9), start_day_after="DRP-2")
    G.add_item(text="Validation", name='DRP-4', group_name="DRP", duration=timedelta(days=8), start_day_after="DRP-3")
    G.add_item(text="Case study", name='DRP-5', group_name="DRP", duration=timedelta(days=6), start_day_after="DRP-4")

    G.add_group(name="FP", text="Final Phase", color_number=1, start_day_after="DRP-2")
    G.add_item(text="Write paper", name="FP-1", group_name="FP", duration=timedelta(days=32), start_day_after="DRP-2")
    G.add_item(text="Green light", name='FP-2', group_name="FP", duration=timedelta(days=1), start_day_after="FP-1")
    G.add_item(text="Thesis write", name='FP-3', group_name="FP", duration=timedelta(days=10), start_day_after="FP-2")
    G.add_item(text="Hand-in", name='hand-in', group_name="FP", duration=timedelta(days=1), start_day_after="FP-3")
    G.add_item(text="Presentation", name='prep-presentation', group_name="FP", duration=timedelta(days=10), start_day_after="hand-in")
    G.add_item(text="Defence", name='FP-5', group_name="FP", duration=timedelta(days=1), start_day_after="prep-presentation")
    print(G.to_latex())
