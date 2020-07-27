import os
import json

import pdb
from IPython.core import ultratb
import sys
sys.excepthook = ultratb.FormattedTB(call_pdb=True)


COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
MATERIALS = ['metal', 'rubber']
SHAPES = ['sphere', 'cylinder', 'cube']


class Executor():
    """Symbolic program executor for V-CLEVR questions"""

    def __init__(self, sim):
        self._set_sim(sim)
        self._register_modules()

    def run(self, pg, debug=False):
        exe_stack = []
        for m in pg:
            #if m=="unique":
            #    import pdb
           #     pdb.set_trace()
            if m in ['<END>', '<NULL>']:
                break
            if m not in ['<START>']:
                if m not in self.modules:
                    exe_stack.append(m)
                else:
                    argv = []
                    for i in range(self.modules[m]['nargs']):
                        if exe_stack:
                            argv.insert(0, exe_stack.pop())
                        else:
                            return 'error'
                    
                    step_output = self.modules[m]['func'](*argv)
                    if step_output == 'error':
                        #pdb.set_trace()
                        return 'error'
                    exe_stack.append(step_output)

                    if debug:
                        print('> %s%s' % (m, argv))
                        print(exe_stack)
        return str(exe_stack[-1])

    def _set_sim(self, sim):
        self.sim = sim
        self.all_objs = sim.get_visible_objs()
        self.existing_events = self._get_events(sim)
        self.unseens = self._get_unseen_events(sim)
        self.causal_traces = self._get_causal_traces(self.all_objs, self.existing_events)

    def _get_events(self, sim, drop_idx=None):
        events = [
            {
                'type': 'start',
                'frame': 0,
            },
            {
                'type': 'end',
                'frame': 125,
            },
        ]
        for io in sim.in_out:
            if io['frame'] < sim.n_vis_frames:
                io_event = {
                    'type': io['type'],
                    'object': io['object'],
                    'frame': io['frame'],
                }
                if drop_idx is not None:
                    io_event = self._convert_event_idx_cf2gt(io_event, drop_idx)
                events.append(io_event)
        for c in sim.collisions:
            if c['frame'] < sim.n_vis_frames:
                col_event = {
                    'type': 'collision',
                    'object': c['object'],
                    'frame': c['frame']
                }
                if drop_idx is not None:
                    col_event = self._convert_event_idx_cf2gt(col_event, drop_idx)
                events.append(col_event)
        return events

    def _get_unseen_events(self, sim):
        """Return a list of events (time_indicators) of events that are 
        going to happen
        """
        unseen_events = []
        for io in self.sim.in_out:
            if io['frame'] >= self.sim.n_vis_frames:
                io_event = {
                    'type': io['type'],
                    'object': [io['object']],
                    'frame': io['frame'],
                }
                unseen_events.append(io_event)
        for c in self.sim.collisions:
            if c['frame'] >= self.sim.n_vis_frames:
                col_event = {
                    'type': 'collision',
                    'object': c['object'],
                    'frame': c['frame'],
                }
                unseen_events.append(col_event)
        return unseen_events

    def _get_causal_traces(self, objs, events):
        """Compute the causal traces for each object"""
        causal_traces = [[] for o in objs]
        for e in self.existing_events:
            if e['type'] not in ['start', 'end']:
                for o in e['object']:
                    causal_traces[o].append(e)
        for i, tr in enumerate(causal_traces):
            tr = sorted(tr, key=lambda k: k['frame'])
            causal_traces[i] = tr
        return causal_traces

    def _convert_event_idx_cf2gt(self, event, drop_idx):
        event_objs_converted = []
        for o in event['object']:
            if o >= drop_idx:
                event_objs_converted.append(o+1)
            else:
                event_objs_converted.append(o)
        event['object'] = event_objs_converted
        return event

    def _register_modules(self):
        self.modules = {
            'objects': {'func': self.objects, 'nargs': 0},
            'events': {'func': self.events, 'nargs': 0},
            'unique': {'func': self.unique, 'nargs': 1},
            'count': {'func': self.count, 'nargs': 1},
            'exist': {'func': self.exist, 'nargs': 1},
            'negate': {'func': self.negate, 'nargs': 1},
            'belong_to': {'func': self.belong_to, 'nargs': 2},
            'filter_color': {'func': self.filter_color, 'nargs': 2},
            'filter_material': {'func': self.filter_material, 'nargs': 2},
            'filter_shape': {'func': self.filter_shape, 'nargs': 2},
            'filter_resting': {'func': self.filter_resting, 'nargs': 2},
            'filter_moving': {'func': self.filter_moving, 'nargs': 2},
            'filter_stationary': {'func': self.filter_stationary, 'nargs': 2},
            'filter_start': {'func': self.filter_start, 'nargs': 1},
            'filter_end': {'func': self.filter_end, 'nargs': 1},
            'filter_in': {'func': self.filter_in, 'nargs': 2},
            'filter_out': {'func': self.filter_out, 'nargs': 2},
            'filter_collision': {'func': self.filter_collision, 'nargs': 2},
            'filter_order': {'func': self.filter_order, 'nargs': 2},
            'filter_before': {'func': self.filter_before, 'nargs': 2},
            'filter_after': {'func': self.filter_after, 'nargs': 2},
            'query_color': {'func': self.query_color, 'nargs': 1},
            'query_material': {'func': self.query_material, 'nargs': 1},
            'query_shape': {'func': self.query_shape, 'nargs': 1},
            'query_direction': {'func': self.query_direction, 'nargs': 2},
            'query_frame': {'func': self.query_frame, 'nargs': 1},
            'query_object': {'func': self.query_object, 'nargs': 1},
            'query_collision_partner': {'func': self.query_collision_partner, 'nargs': 2},
            'filter_ancestor': {'func': self.filter_ancestor, 'nargs': 2},
            'unseen_events': {'func': self.unseen_events, 'nargs': 0},
            'all_events': {'func': self.all_events, 'nargs': 0},
            'counterfact_events': {'func': self.counterfact_events, 'nargs': 1},
            'filter_counterfact': {'func': self.filter_counterfact, 'nargs': 2},
        }

    # Module definitions

    ## Set / entry operators
    def objects(self):
        """
        Return full object list
        - args:
        - return: objects(list)
        """
        return self.all_objs[:]

    def events(self):
        """
        Return full event list sorted in time order
        - args:
        - return: events(list)
        """
        events = self.existing_events[:]
        events = sorted(events, key=lambda k: k['frame'])
        return events

    def unique(self, input_list):
        """
        Return the only element of a list
        - args: objects / events (list)
        - return: object / event
        """
        if type(input_list) is not list:
            return 'error'
        if len(input_list) != 1:
            return 'error'
        else:
            return input_list[0]

    def count(self, input_list):
        """
        Return the number of objects / events in the input list
        - args: objects / events (list)
        - return: count(int)
        """
        if type(input_list) is not list:
            return 'error'
        return len(input_list)

    def exist(self, input_list):
        """
        Return if the input list is not empty
        - args: objects / events (list)
        - return: (bool)
        """
        if type(input_list) is not list:
            return 'error'
        if len(input_list) > 0:
            return 'yes'
        else:
            return 'no'

    def negate(self, input_bool):
        """
        Negate the input yes / no statement
        - args: input_bool(str)
        - return: output_bool(str)
        """
        if input_bool == 'yes':
            return 'no'
        if input_bool == 'no':
            return 'yes'
        return 'error'

    def belong_to(self, input_entry, events):
        """
        Return if the input event / object belongs to the event list
        - args: input_entry(dict / int), events(list)
        - return: output_bool(str)
        """
        if type(events) is not list:
            return 'error'
        if type(input_entry) is dict:
            for e in events:
                if e['type'] not in ['start', 'end']:
                    if input_entry['type'] == e['type'] and \
                       set(input_entry['object']) == set(e['object']):
                        return 'yes'
            return 'no'
        elif type(input_entry) is int:
            for e in events:
                if e['type'] not in ['start', 'end']:
                    if input_entry in e['object']:
                        return 'yes'
            return 'no'
        else:
            return 'error'

    ## Object filters
    def filter_color(self, objs, color):
        """
        Filter objects by color
        - args: objects(list), color(str)
        - return: objects(list)
        """
        if type(objs) is not list:
            return 'error'
        if len(objs) > 0 and type(objs[0]) is not int:
            return 'error'
        if color not in COLORS:
            return 'error'
        output_objs = []
        for o in objs:
            obj_attr = self.sim.get_static_attrs(o)
            if obj_attr['color'] == color:
                output_objs.append(o)
        return output_objs

    def filter_material(self, objs, material):
        """
        Filter objects by material
        - args: objects(list), material(str)
        - return: objects(list)
        """
        if type(objs) is not list:
            return 'error'
        if len(objs) > 0 and type(objs[0]) is not int:
            return 'error'
        if material not in MATERIALS:
            return 'error'
        output_objs = []
        for o in objs:
            obj_attr = self.sim.get_static_attrs(o)
            if obj_attr['material'] == material:
                output_objs.append(o)
        return output_objs

    def filter_shape(self, objs, shape):
        """
        Filter objects by shape
        - args: objects(list), shape(str)
        - return: objects(list)
        """
        if type(objs) is not list:
            return 'error'
        if len(objs) > 0 and type(objs[0]) is not int:
            return 'error'
        if shape not in SHAPES:
            return 'error'
        output_objs = []
        for o in objs:
            obj_attr = self.sim.get_static_attrs(o)
            if obj_attr['shape'] == shape:
                output_objs.append(o)
        return output_objs

    def filter_resting(self, objs, frame):
        """
        Filter all resting objects in the input list
        - args: objects(list), frame(int)
        - return: objects(list)
        """
        if type(objs) is not list:
            return 'error'
        if type(frame) is not int and frame != 'null':
            return 'error'
        if frame == 'null':
            frame = None
        output_objs = []
        for o in objs:
            if self.sim.is_visible(o, frame_idx=frame) \
               and not self.sim.is_moving(o, frame_idx=frame):
                output_objs.append(o)
        return output_objs

    def filter_moving(self, objs, frame):
        """
        Filter all moving objects in the input list
        - args: objects(list), frame(int)
        - return: objects(list)
        """
        if type(objs) is not list:
            return 'error'
        if type(frame) is not int and frame != 'null':
            return 'error'
        if frame == 'null':
            frame = None
        output_objs = []
        for o in objs:
            if self.sim.is_visible(o, frame_idx=frame) and \
               self.sim.is_moving(o, frame_idx=frame):
                output_objs.append(o)
        return output_objs

    def filter_stationary(self, objs, frame):
        """
        Filter all moving objects in the input list
        - args: objects(list), frame(int)
        - return: objects(list)
        """
        if type(objs) is not list:
            return 'error'
        if type(frame) is not int and frame != 'null':
            return 'error'
        if frame == 'null':
            frame = None
        output_objs = []
        for o in objs:
            if self.sim.is_visible(o, frame_idx=frame) and \
               not self.sim.is_moving(o, frame_idx=frame):
                output_objs.append(o)
        return output_objs

    ## Event filters
    def filter_start(self, events):
        """
        Find and return the start event from input list
        - args: events(list)
        - return: event(dict)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        for e in events:
            if e['type'] == 'start':
                return e
        return 'error'

    def filter_end(self, events):
        """
        Find and return the end event from input list
        - args: events(list)
        - return: event(dict)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        for e in events:
            if e['type'] == 'end':
                return e
        return error

    def filter_in(self, events, objs):
        """
        Return all incoming events that involve any of the objects
        args: events(list), objects(list / int)
        return: events(list)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        if type(objs) is not list:
            objs = [objs]
        if len(objs) > 0 and type(objs[0]) is not int:
            return 'error'
        output_events = []
        for e in events:
            if e['type'] == 'in':
                if e['object'][0] in objs:
                    output_events.append(e)
        return output_events

    def filter_out(self, events, objs):
        """
        Return all outgoing events that involve any of the objects
        args: events(list), objects(list / int)
        return: events(list)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        if type(objs) is not list:
            objs = [objs]
        if len(objs) > 0 and type(objs[0]) is not int:
            return 'error'
        output_events = []
        for e in events:
            if e['type'] == 'out':
                if e['object'][0] in objs:
                    output_events.append(e)
        return output_events

    def filter_collision(self, events, objs):
        """
        Return all collision events that involve any of the objects
        args: events(list), objects(list / int)
        return: events(list)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        if type(objs) is not list:
            objs = [objs]
        if len(objs) > 0 and type(objs[0]) is not int:
            return 'error'
        output_events = []
        for e in events:
            if e['type'] == 'collision':
                if e['object'][0] in objs or e['object'][1] in objs:
                    output_events.append(e)
        return output_events

    def filter_order(self, events, order):
        """
        Return the event at given order
        - args: events(list), order(str)
        - return: event(dict)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        if order not in ['first', 'second', 'last']:
            return 'error'
        idx_dict = {
            'first': 0,
            'second': 1,
            'last': -1
        }
        idx = idx_dict[order]
        if idx >= len(events) or len(events) == 0:
            return 'error'
        return events[idx]

    def filter_before(self, events, event):
        """
        Return all events before the designated event
        - args: events(list), event(dict)
        - return: events(list)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        if 'type' not in event:
            return 'error'
        output_events = []
        for e in events:
            if e['frame'] < event['frame']:
                output_events.append(e)
        return output_events

    def filter_after(self, events, event):
        """
        Return all events after the designated event
        - args: events(list), event(dict)
        - return: events(list)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        if type(event) is not dict or 'type' not in event:
            return 'error'
        output_events = []
        for e in events:
            if e['frame'] > event['frame']:
                output_events.append(e)
        return output_events

    ## query objects
    def query_color(self, obj):
        """
        Return the color of the queried object
        - args: obj(int)
        - return: color(str)
        """
        if type(obj) is not int:
            return 'error'
        obj_attr = self.sim.get_static_attrs(obj)
        return obj_attr['color']

    def query_material(self, obj):
        """
        Return the material of the queried object
        - args: obj(int)
        - return material(str)
        """
        if type(obj) is not int:
            return 'error'
        obj_attr = self.sim.get_static_attrs(obj)
        return obj_attr['material']

    def query_shape(self, obj):
        """
        Return the shape of the queried object
        - args: obj(int)
        - return: shape(str)
        """
        if type(obj) is not int:
            return 'error'
        obj_attr = self.sim.get_static_attrs(obj)
        return obj_attr['shape']

    def query_direction(self, obj, frame):
        """
        Return the direction of the queried object
        - args: obj(int), frame(int)
        - return: direction(str)
        """
        if type(obj) is not int or type(frame) is not int:
            return 'error'
        if self.sim.is_moving_left(obj, frame, angle_half_range=40):
            return 'left'
        if self.sim.is_moving_right(obj, frame, angle_half_range=40):
            return 'right'
        if self.sim.is_moving_up(obj, frame, angle_half_range=40):
            return 'up'
        if self.sim.is_moving_down(obj, frame, angle_half_range=40):
            return 'down'
        return 'error'

    ## query events
    def query_frame(self, event):
        """
        Return the frame number of the queried event
        - args: event(dict)
        - return: frame(int)
        """
        if type(event) is not dict:
            return 'error'
        return event['frame']

    def query_object(self, in_out_event):
        """
        Return the object index that involves in the event
        - args: event(dict)
        - return: obj(int)
        """
        if type(in_out_event) is not dict:
            return 'error'
        if in_out_event['type'] not in ['in', 'out']:
            return 'error'
        return in_out_event['object'][0]

    def query_collision_partner(self, col_event, obj):
        """
        Return the collision partner of the input object
        - args: col_event(dict), obj(int)
        - return: obj(int)
        """
        if type(col_event) is not dict or type(obj) is not int:
            return 'error'
        if col_event['type'] != 'collision':
            return 'error'
        if obj not in col_event['object']:
            return 'error'
        col_objs = col_event['object']
        if obj == col_objs[0]:
            return col_objs[1]
        else:
            return col_objs[0]

    ## Explanatory
    def _search_causes(self, target_event, causes):
        """Recursively search for all ancestor events of the target event in the causal graph"""
        next_step_events = []
        for tr in self.causal_traces:
            if target_event in tr:
                idx = tr.index(target_event)
                if idx > 0 and tr[idx-1] not in next_step_events:
                    next_step_events.append(tr[idx-1])
                    if tr[idx-1] not in causes:
                        causes.append(tr[idx-1])
        for e in next_step_events:
            self._search_causes(e, causes)

    def filter_ancestor(self, events, event):
        """
        Filter all ancestors of the input event in the causal graph
        - args: events(list), event(dict)
        - return: events(list)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        if type(event) is not dict or 'type' not in event:
            return 'error'
        all_causes = []
        self._search_causes(event, all_causes)
        output = []
        for e in events:
            if e in all_causes:
                output.append(e)
        return output

    ## Predictive
    def unseen_events(self):
        """
        Return a complete list of all unseen events
        - args: 
        - return: events(list)
        """
        return self.unseens

    ## Counterfactual
    def all_events(self):
        """
        Return all possible events
        - args:
        - return: events(list)
        """
        possible_events = []
        for i in range(len(self.all_objs)):
            for j in range(i+1, len(self.all_objs)):
                event = {
                    'type': 'collision',
                    'object': [self.all_objs[i], self.all_objs[j]],
                }
                possible_events.append(event)
        return possible_events

    def counterfact_events(self, remove_obj):
        """
        Return all events after removing the input object
        - args: remove_obj(int)
        - return: counterfact_events(list)
        """
        if type(remove_obj) is not int:
            return 'error'
        if remove_obj not in self.all_objs:
            return 'error'
        return self.sim.cf_events[remove_obj]

    def filter_counterfact(self, events, remove_obj):
        """
        Return all events from the input list that will happen when the object is dropped
        - args: events(list), remove_obj(int)
        - return: events(list)
        """
        if type(events) is not list:
            return 'error'
        if len(events) > 0 and type(events[0]) is not dict:
            return 'error'
        cf_events = self.counterfact_events(remove_obj)
        if cf_events == 'error':
            return 'error'
        cf_col_pairs = [set(e['object']) for e in cf_events if e['type'] == 'collision']
        outputs = []
        for e in events:
            if set(e['object']) in cf_col_pairs:
                outputs.append(e)
        return outputs
